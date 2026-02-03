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

#include "pl/Hook.h"
#include "pl/Gloss.h"

// =============================================================
// 1. ZENITH SETTINGS
// =============================================================
static const float SCALE = 0.5f;          // 50% Res (Max FPS)
static const float BASE_BLUR = 0.86f;     // 86% Smoothness (Liquid Feel)
static const float NEON_MAX = 0.98f;      // 98% Light Trails
static const float SHARPEN_STR = 0.90f;   // 90% Clarity (Ultra HD)

// =============================================================
// 2. SHADERS (Cinema Grade)
// =============================================================

const char* vert = R"(#version 300 es
layout(location=0) in vec4 p; layout(location=1) in vec2 t; out mediump vec2 v;
void main(){gl_Position=p;v=t;})";

// --- PASS 1: SMART ACCUMULATION ---
const char* frag_neon = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D c;
uniform sampler2D h;
out vec4 o;

// Interleaved Gradient Noise (Cinematic Dithering)
float interleavedGradientNoise(vec2 uv) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

void main() {
    lowp vec4 curr = texture(c, v);
    
    // 1. CENTER MASK
    mediump vec2 center = vec2(0.5);
    lowp float dist = distance(v, center);
    lowp float mask = smoothstep(0.05, 0.30, dist);

    // 2. WARP TUNNEL
    mediump vec2 zoomUV = (v - center) * 0.993 + center;
    lowp vec4 hist = texture(h, zoomUV);

    // 3. VELOCITY PROTECTION (Anti-Ghosting)
    // Calculate how different the new frame is from history.
    // If difference is huge (Fast Flick), reduce blur to prevent ghosting.
    lowp float diff = distance(curr.rgb, hist.rgb);
    lowp float ghostFix = 1.0 - smoothstep(0.2, 0.6, diff); 

    // 4. HYBRID BLUR LOGIC
    lowp float luma = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    lowp float boost = smoothstep(0.4, 0.9, luma) * 0.12;
    
    // Combine: Base Blur + Light Boost + Velocity Fix
    lowp float decay = (0.86 + boost) * ghostFix;

    // 5. COLOR PRESERVATION + DITHER
    lowp float dither = (interleavedGradientNoise(gl_FragCoord.xy) - 0.5) * 0.015;
    
    lowp vec4 finalHist = hist;
    if (luma > 0.6) {
        finalHist = hist * 1.01 + dither; // Glow
    } else {
        finalHist = hist + dither;        // Clean
    }

    o = mix(curr, finalHist, decay * mask);
})";

// --- PASS 2: VISUAL FIDELITY ---
const char* frag_draw = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D t;
uniform highp float uTime;
out vec4 o;

void main() {
    // 1. HEAT HAZE
    lowp vec4 rawCol = texture(t, v);
    lowp float luma = dot(rawCol.rgb, vec3(0.299, 0.587, 0.114));
    lowp float heatMask = smoothstep(0.75, 1.0, luma);

    highp float time = uTime;
    mediump vec2 ripple = vec2(sin(v.y*25.0+time), cos((v.x+v.y)*20.0+time*1.2)) * 0.002;
    mediump vec2 finalUV = v + (ripple * heatMask);

    // 2. ADAPTIVE CAS SHARPENING
    lowp vec4 col = texture(t, finalUV);
    lowp vec4 n = textureOffset(t, finalUV, ivec2(0, -1));
    lowp vec4 s = textureOffset(t, finalUV, ivec2(0, 1));
    lowp vec4 e = textureOffset(t, finalUV, ivec2(1, 0));
    lowp vec4 w = textureOffset(t, finalUV, ivec2(-1, 0));

    // Contrast Check
    lowp float mx = max(col.g, max(max(n.g, s.g), max(e.g, w.g)));
    lowp float mn = min(col.g, min(min(n.g, s.g), min(e.g, w.g)));
    lowp float amt = sqrt(clamp(mn / (1.0 - mx + 0.001), 0.0, 1.0));

    // DARKNESS PROTECTION:
    // If pixel is dark (Cave), reduce sharpening strength to 0.2.
    // If pixel is bright (Surface), use full strength 0.9.
    // This prevents "static noise" in dark areas.
    lowp float darkProtect = smoothstep(0.0, 0.3, luma); 
    lowp float finalStr = 0.90 * darkProtect;

    lowp float peak = -1.0 / mix(8.0, 5.0, amt * finalStr); 
    
    lowp vec4 sharp = col + (n + s + e + w) * peak;
    col = sharp / (1.0 + 4.0 * peak);

    // 3. DYNAMIC EXPOSURE
    col.rgb += (1.0 - smoothstep(0.0, 0.4, luma)) * 0.15;

    // 4. ACES TONEMAP
    lowp vec3 x = col.rgb;
    col.rgb = clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);

    o = col;
})";

// =============================================================
// 3. RENDER ENGINE
// =============================================================
static GLuint rawTex=0, rawFBO=0, histTex[2]={0,0}, histFBO[2]={0,0}, vao=0;
static GLuint progNeon=0, progDraw=0;
static int ping=0, iW=0, iH=0, sW=0, sH=0;

float getTime() {
    struct timespec res; clock_gettime(CLOCK_MONOTONIC, &res);
    return (float)(res.tv_sec) + (float)(res.tv_nsec) / 1e9;
}

void initGL(int w, int h) {
    if(rawTex){glDeleteTextures(1,&rawTex); glDeleteFramebuffers(1,&rawFBO); glDeleteTextures(2,histTex); glDeleteFramebuffers(2,histFBO); glDeleteVertexArrays(1,&vao);}
    iW=(int)(w*SCALE); iH=(int)(h*SCALE);
    auto c=[](GLenum t, const char* s){GLuint x=glCreateShader(t); glShaderSource(x,1,&s,0); glCompileShader(x); return x;};
    GLuint vs=c(GL_VERTEX_SHADER,vert), fs1=c(GL_FRAGMENT_SHADER,frag_neon), fs2=c(GL_FRAGMENT_SHADER,frag_draw);
    
    progNeon=glCreateProgram(); glAttachShader(progNeon,vs); glAttachShader(progNeon,fs1); glLinkProgram(progNeon);
    glUseProgram(progNeon); glUniform1i(glGetUniformLocation(progNeon,"c"),0); glUniform1i(glGetUniformLocation(progNeon,"h"),1);

    progDraw=glCreateProgram(); glAttachShader(progDraw,vs); glAttachShader(progDraw,fs2); glLinkProgram(progDraw);
    
    GLfloat d[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, 1,1,1,1}; GLushort i[]={0,1,2, 0,2,3};
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    GLuint vb,ib; glGenBuffers(1,&vb); glBindBuffer(GL_ARRAY_BUFFER,vb); glBufferData(GL_ARRAY_BUFFER,sizeof(d),d,GL_STATIC_DRAW);
    glGenBuffers(1,&ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(i),i,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,0,16,0); glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,0,16,(void*)8);

    auto t = [&](GLuint& tx, GLuint& fb){
        glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_2D,tx);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,iW,iH,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tx,0);
    };
    t(rawTex,rawFBO); t(histTex[0],histFBO[0]); t(histTex[1],histFBO[1]);
    
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[0]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[1]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    sW=w; sH=h;
}

void render(int w, int h) {
    if(w!=sW || h!=sH || !rawTex) initGL(w,h);
    GLint lastFBO; glGetIntegerv(GL_FRAMEBUFFER_BINDING,&lastFBO);
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. DOWNSCALE
    glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,rawFBO);
    glBlitFramebuffer(0,0,w,h,0,0,iW,iH,GL_COLOR_BUFFER_BIT,GL_LINEAR);

    // 2. PASS 1
    int cur=ping, pre=1-ping;
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[cur]); glViewport(0,0,iW,iH); glBindVertexArray(vao);
    glUseProgram(progNeon);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,rawTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,histTex[pre]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    // 3. PASS 2
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,w,h);
    glUseProgram(progDraw);
    glUniform1f(glGetUniformLocation(progDraw, "uTime"), getTime() * 2.8);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,histTex[cur]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    ping=pre;
}

EGLBoolean (*orig)(EGLDisplay,EGLSurface)=0;
EGLBoolean hook(EGLDisplay d, EGLSurface s){
    EGLint w,h; eglQuerySurface(d,s,EGL_WIDTH,&w); eglQuerySurface(d,s,EGL_HEIGHT,&h);
    if(w>100) render(w,h);
    return orig(d,s);
}
void* mainthread(void*){sleep(1); GlossInit(true); GlossHook((void*)GlossSymbol(GlossOpen("libEGL.so"),"eglSwapBuffers",0),(void*)hook,(void**)&orig); return 0;}
__attribute__((constructor)) void init(){pthread_t t;pthread_create(&t,0,mainthread,0);}
