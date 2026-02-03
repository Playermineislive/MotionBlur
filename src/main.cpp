#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES3/gl3.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cmath>
#include <time.h>

// Preloader Headers
#include "pl/Hook.h"
#include "pl/Gloss.h"

// =============================================================
// 1. SETTINGS (Calibrated for "Apex" Quality)
// =============================================================
static const float SCALE = 0.5f;          // 50% Resolution (Foundation of FPS)
static const float HAZE_STR = 0.002f;     // Heat Distortion Amount (Subtle is better)
static const float HAZE_SPEED = 2.8f;     // Speed of heat ripples
static const float SHARPEN_STR = 0.70f;   // CAS Sharpening Strength (0.0 - 1.0)
static const float TRAIL_DECAY = 0.96f;   // Light Trail Length

// =============================================================
// 2. SHADERS (Verified Logic)
// =============================================================

const char* vert = R"(#version 300 es
layout(location=0) in vec4 p; layout(location=1) in vec2 t; out mediump vec2 v;
void main(){gl_Position=p;v=t;})";

// --- PASS 1: NEON ENGINE ---
// Handles: Light Trails, Peripheral Warp, Center Masking
const char* frag_neon = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D c; // Current Frame
uniform sampler2D h; // History Frame
out vec4 o;

void main() {
    lowp vec4 curr = texture(c, v);
    
    // 1. CENTER MASK
    // Calculates distance from center (0.5).
    // Protects the crosshair (radius 0.15) from blurring.
    mediump vec2 center = vec2(0.5);
    lowp float dist = distance(v, center);
    lowp float mask = smoothstep(0.12, 0.45, dist);

    // 2. WARP TUNNEL
    // Reads history from a slightly "zoomed in" coordinate.
    // Creates the sensation of speed/motion at the edges.
    mediump vec2 zoomUV = (v - center) * 0.994 + center;
    lowp vec4 hist = texture(h, zoomUV);

    // 3. BRIGHTNESS DECAY
    // Calculates Luma (Perceived Brightness).
    lowp float luma = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    
    // Logic: Only retain history if the pixel is bright (Luma > 0.4).
    // Dark pixels fade instantly to avoid "muddy" graphics.
    lowp float decay = smoothstep(0.4, 0.9, luma) * 0.96;

    // 4. MIX
    // If mask=0 (Center), output Current.
    // If mask=1 (Edge), mix Current + History.
    o = mix(curr, hist * 1.01, decay * mask);
})";

// --- PASS 2: VISUAL ENGINE ---
// Handles: Heat Haze, CAS Sharpening, Auto-Exposure, Tonemapping
const char* frag_draw = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D t;
uniform highp float uTime;
out vec4 o;

void main() {
    // 1. HEAT HAZE MASK
    // Determine brightness of the UNDISTORTED pixel.
    lowp vec4 rawCol = texture(t, v);
    lowp float luma = dot(rawCol.rgb, vec3(0.299, 0.587, 0.114));
    lowp float heatMask = smoothstep(0.7, 1.0, luma);

    // 2. GENERATE DISTORTION
    highp float time = uTime;
    // Interleaved Sin/Cos waves for organic fluid motion.
    mediump vec2 ripple = vec2(
        sin(v.y * 30.0 + time), 
        cos((v.x + v.y) * 25.0 + time * 1.3)
    ) * 0.002; // Strength

    // Apply distortion only to hot pixels
    mediump vec2 finalUV = v + (ripple * heatMask);

    // 3. CAS SHARPENING (Applied to Distorted UV)
    // We read the texture at 'finalUV'.
    lowp vec4 col = texture(t, finalUV);
    
    // We read neighbors relative to 'finalUV' to ensure edges align with distortion.
    lowp vec4 n = textureOffset(t, finalUV, ivec2(0, -1));
    lowp vec4 s = textureOffset(t, finalUV, ivec2(0, 1));
    lowp vec4 e = textureOffset(t, finalUV, ivec2(1, 0));
    lowp vec4 w = textureOffset(t, finalUV, ivec2(-1, 0));

    // Contrast calculation (Green Channel Luma Approximation)
    lowp float mx = max(col.g, max(max(n.g, s.g), max(e.g, w.g)));
    lowp float mn = min(col.g, min(min(n.g, s.g), min(e.g, w.g)));
    
    // Sharpening math (AMD FSR Logic adapted for Mobile)
    lowp float amt = sqrt(clamp(mn / (1.0 - mx + 0.001), 0.0, 1.0));
    lowp float peak = -1.0 / mix(8.0, 5.0, amt * 0.7); // 0.7 = Sharpen Strength
    
    lowp vec4 sharp = col + (n + s + e + w) * peak;
    col = sharp / (1.0 + 4.0 * peak);

    // 4. DYNAMIC EXPOSURE (Auto-Brightness)
    // Boosts shadows slightly (0.12) to simulate eye adaptation.
    col.rgb += (1.0 - smoothstep(0.0, 0.4, luma)) * 0.12;

    // 5. ACES TONEMAPPING
    // Standard filmic curve for contrast punch.
    lowp vec3 x = col.rgb;
    col.rgb = clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);

    o = col;
})";

// =============================================================
// 3. RENDER LOOP
// =============================================================
static GLuint rawTex=0, rawFBO=0, histTex[2]={0,0}, histFBO[2]={0,0}, vao=0;
static GLuint progNeon=0, progDraw=0;
static int ping=0, iW=0, iH=0, sW=0, sH=0;

// High-precision timer for Heat Haze animation
float getTime() {
    struct timespec res; clock_gettime(CLOCK_MONOTONIC, &res);
    return (float)(res.tv_sec) + (float)(res.tv_nsec) / 1e9;
}

void initGL(int w, int h) {
    // Cleanup old resources if resolution changed
    if(rawTex){glDeleteTextures(1,&rawTex); glDeleteFramebuffers(1,&rawFBO); glDeleteTextures(2,histTex); glDeleteFramebuffers(2,histFBO); glDeleteVertexArrays(1,&vao);}
    
    // Calculate Internal Resolution (Downscaled)
    iW=(int)(w*SCALE); iH=(int)(h*SCALE);

    // Shader Compilation Helper
    auto c=[](GLenum t, const char* s){GLuint x=glCreateShader(t); glShaderSource(x,1,&s,0); glCompileShader(x); return x;};
    GLuint vs=c(GL_VERTEX_SHADER,vert), fs1=c(GL_FRAGMENT_SHADER,frag_neon), fs2=c(GL_FRAGMENT_SHADER,frag_draw);
    
    // Compile Neon Program
    progNeon=glCreateProgram(); glAttachShader(progNeon,vs); glAttachShader(progNeon,fs1); glLinkProgram(progNeon);
    glUseProgram(progNeon); glUniform1i(glGetUniformLocation(progNeon,"c"),0); glUniform1i(glGetUniformLocation(progNeon,"h"),1);

    // Compile Visuals Program
    progDraw=glCreateProgram(); glAttachShader(progDraw,vs); glAttachShader(progDraw,fs2); glLinkProgram(progDraw);
    
    // Setup Fullscreen Quad
    GLfloat d[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, 1,1,1,1}; GLushort i[]={0,1,2, 0,2,3};
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    GLuint vb,ib; glGenBuffers(1,&vb); glBindBuffer(GL_ARRAY_BUFFER,vb); glBufferData(GL_ARRAY_BUFFER,sizeof(d),d,GL_STATIC_DRAW);
    glGenBuffers(1,&ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(i),i,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,0,16,0); glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,0,16,(void*)8);

    // Setup Textures (Low Res for performance)
    auto t = [&](GLuint& tx, GLuint& fb){
        glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_2D,tx);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,iW,iH,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tx,0);
    };
    t(rawTex,rawFBO); t(histTex[0],histFBO[0]); t(histTex[1],histFBO[1]);
    
    // Clear History Buffers (Avoids visual garbage on startup)
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[0]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[1]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    sW=w; sH=h;
}

void render(int w, int h) {
    if(w!=sW || h!=sH || !rawTex) initGL(w,h);
    
    GLint lastFBO; glGetIntegerv(GL_FRAMEBUFFER_BINDING,&lastFBO);
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. FAST COPY (Screen -> LowRes Texture)
    glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,rawFBO);
    glBlitFramebuffer(0,0,w,h,0,0,iW,iH,GL_COLOR_BUFFER_BIT,GL_LINEAR);

    // 2. PASS 1: NEON ENGINE
    int cur=ping, pre=1-ping;
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[cur]); glViewport(0,0,iW,iH); glBindVertexArray(vao);
    glUseProgram(progNeon);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,rawTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,histTex[pre]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    // 3. PASS 2: VISUAL ENGINE
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,w,h);
    glUseProgram(progDraw);
    glUniform1f(glGetUniformLocation(progDraw, "uTime"), getTime() * HAZE_SPEED);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,histTex[cur]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    ping=pre;
}

// =============================================================
// 4. HOOKS
// =============================================================
EGLBoolean (*orig)(EGLDisplay,EGLSurface)=0;
EGLBoolean hook(EGLDisplay d, EGLSurface s){
    EGLint w,h; eglQuerySurface(d,s,EGL_WIDTH,&w); eglQuerySurface(d,s,EGL_HEIGHT,&h);
    if(w>100) render(w,h);
    return orig(d,s);
}

void* mainthread(void*){
    sleep(1); GlossInit(true);
    GHandle h = GlossOpen("libEGL.so");
    void* s = (void*)GlossSymbol(h,"eglSwapBuffers",0);
    if(s) GlossHook(s, (void*)hook, (void**)&orig);
    return 0;
}

__attribute__((constructor)) void init(){pthread_t t;pthread_create(&t,0,mainthread,0);}
