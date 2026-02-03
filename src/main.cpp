#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES3/gl3.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cmath>

#include "pl/Hook.h"
#include "pl/Gloss.h"

// =============================================================
// 1. FINAL SETTINGS
// =============================================================
static const float SCALE = 0.5f;          // 50% Internal Resolution (Max Performance)
static const float MAX_BLUR = 0.94f;      // 94% Smoothness (Walking/Looking around)
static const float MIN_BLUR = 0.35f;      // 35% Smoothness (Fast PvP Flicks)
static const float SHARPEN = 0.88f;       // 88% CAS Sharpening (HD Clarity)

// =============================================================
// 2. SHADERS (Verified & Optimized)
// =============================================================

const char* vert = R"(#version 300 es
layout(location=0) in vec4 p; layout(location=1) in vec2 t; out mediump vec2 v;
void main(){gl_Position=p;v=t;})";

// --- PASS 1: VELOCITY ACCUMULATION ---
const char* frag_blur = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D c; // Current Frame
uniform sampler2D h; // History Frame
out vec4 o;

void main() {
    lowp vec4 curr = texture(c, v);
    lowp vec4 hist = texture(h, v);

    // 1. VELOCITY CALCULATOR (Anti-Ghosting)
    lowp float lC = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lH = dot(hist.rgb, vec3(0.299, 0.587, 0.114));
    lowp float diff = abs(lC - lH);

    // Dynamic Interpolation:
    // Low Diff (Walking) -> Max Blur (0.94)
    // High Diff (Flicking) -> Min Blur (0.35)
    lowp float velocity = smoothstep(0.02, 0.30, diff);
    lowp float factor = mix(0.94, 0.35, velocity);

    // 2. SHADOW PROTECTION (Contrast Fix)
    // If history is darker than current, we favor it slightly.
    // This prevents shadows from turning gray during movement.
    lowp vec4 result = mix(curr, hist, factor);
    if (lH < lC) { 
        result = mix(result, hist, 0.05); 
    }

    // 3. CENTER MASK (PvP Aim)
    // Protects the crosshair area (Radius 0.12)
    mediump vec2 center = vec2(0.5);
    lowp float dist = distance(v, center);
    lowp float mask = smoothstep(0.01, 0.12, dist);

    o = mix(curr, result, mask);
})";

// --- PASS 2: CLARITY & OUTPUT ---
const char* frag_draw = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D t;
out vec4 o;

void main() {
    // 1. CAS SHARPENING (Contrast Adaptive Sharpening)
    lowp vec4 col = texture(t, v);
    
    // Read 4 neighbors
    lowp vec4 n = textureOffset(t, v, ivec2(0, -1));
    lowp vec4 s = textureOffset(t, v, ivec2(0, 1));
    lowp vec4 e = textureOffset(t, v, ivec2(1, 0));
    lowp vec4 w = textureOffset(t, v, ivec2(-1, 0));

    // Calculate Luma for cheap/fast processing
    lowp float lC = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lN = dot(n.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lS = dot(s.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lE = dot(e.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lW = dot(w.rgb, vec3(0.299, 0.587, 0.114));

    // Calculate Contrast
    lowp float mx = max(lC, max(max(lN, lS), max(lE, lW)));
    lowp float mn = min(lC, min(min(lN, lS), min(lE, lW)));
    lowp float amt = sqrt(clamp(mn / (1.0 - mx + 0.001), 0.0, 1.0));
    
    // Apply Sharpening (Strength 0.88)
    lowp float peak = -1.0 / mix(8.0, 5.0, amt * 0.88); 
    lowp float sharpLuma = lC + (lN + lS + lE + lW) * peak;
    sharpLuma /= (1.0 + 4.0 * peak);
    
    // Apply Luma delta to Color
    col.rgb += (sharpLuma - lC);

    // 2. VIBRANCE (Color Restoration)
    // Boosts muted colors slightly to counter blur washout
    lowp float maxRGB = max(col.r, max(col.g, col.b));
    lowp float minRGB = min(col.r, min(col.g, col.b));
    lowp float sat = maxRGB - minRGB;
    col.rgb = mix(col.rgb, vec3(maxRGB), (1.0 - pow(sat, 0.5)) * -0.2);

    // 3. ACES TONEMAP
    lowp vec3 x = col.rgb;
    col.rgb = clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);

    // 4. ALPHA SAFETY (Fixes UI Bugs)
    o = vec4(col.rgb, 1.0);
})";

// =============================================================
// 3. RENDER ENGINE
// =============================================================
static GLuint rawTex=0, rawFBO=0, histTex[2]={0,0}, histFBO[2]={0,0}, vao=0;
static GLuint progBlur=0, progDraw=0;
static int ping=0, iW=0, iH=0, sW=0, sH=0;

void initGL(int w, int h) {
    // Resource cleanup
    if(rawTex){glDeleteTextures(1,&rawTex); glDeleteFramebuffers(1,&rawFBO); glDeleteTextures(2,histTex); glDeleteFramebuffers(2,histFBO); glDeleteVertexArrays(1,&vao);}
    
    // Internal Resolution
    iW=(int)(w*SCALE); iH=(int)(h*SCALE);

    // Shader Compilation
    auto c=[](GLenum t, const char* s){GLuint x=glCreateShader(t); glShaderSource(x,1,&s,0); glCompileShader(x); return x;};
    GLuint vs=c(GL_VERTEX_SHADER,vert), fs1=c(GL_FRAGMENT_SHADER,frag_blur), fs2=c(GL_FRAGMENT_SHADER,frag_draw);
    
    progBlur=glCreateProgram(); glAttachShader(progBlur,vs); glAttachShader(progBlur,fs1); glLinkProgram(progBlur);
    glUseProgram(progBlur); glUniform1i(glGetUniformLocation(progBlur,"c"),0); glUniform1i(glGetUniformLocation(progBlur,"h"),1);

    progDraw=glCreateProgram(); glAttachShader(progDraw,vs); glAttachShader(progDraw,fs2); glLinkProgram(progDraw);
    
    // Geometry Setup
    GLfloat d[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, 1,1,1,1}; GLushort i[]={0,1,2, 0,2,3};
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    GLuint vb,ib; glGenBuffers(1,&vb); glBindBuffer(GL_ARRAY_BUFFER,vb); glBufferData(GL_ARRAY_BUFFER,sizeof(d),d,GL_STATIC_DRAW);
    glGenBuffers(1,&ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(i),i,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,0,16,0); glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,0,16,(void*)8);

    // Texture Setup
    auto t = [&](GLuint& tx, GLuint& fb){
        glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_2D,tx);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,iW,iH,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tx,0);
    };
    t(rawTex,rawFBO); t(histTex[0],histFBO[0]); t(histTex[1],histFBO[1]);
    
    // Clear Buffers
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[0]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[1]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    sW=w; sH=h;
}

void render(int w, int h) {
    if(w!=sW || h!=sH || !rawTex) initGL(w,h);
    
    // Save state is not strictly required for SwapBuffers hooks on Android, 
    // but disabling tests is crucial for our full-screen pass.
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. FAST COPY (Downscale)
    glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,rawFBO);
    glBlitFramebuffer(0,0,w,h,0,0,iW,iH,GL_COLOR_BUFFER_BIT,GL_LINEAR);

    // 2. BLUR PASS
    int cur=ping, pre=1-ping;
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[cur]); glViewport(0,0,iW,iH); glBindVertexArray(vao);
    glUseProgram(progBlur);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,rawTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,histTex[pre]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    // 3. DRAW PASS (Upscale + Sharpen)
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,w,h);
    glUseProgram(progDraw);
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

__attribute__((constructor)) void init(){pthread_t t;pthread_create(&t,0,mainthread,0);}#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES3/gl3.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cmath>

#include "pl/Hook.h"
#include "pl/Gloss.h"

// =============================================================
// 1. FINAL SETTINGS
// =============================================================
static const float SCALE = 0.5f;          // 50% Internal Resolution (Max Performance)
static const float MAX_BLUR = 0.94f;      // 94% Smoothness (Walking/Looking around)
static const float MIN_BLUR = 0.35f;      // 35% Smoothness (Fast PvP Flicks)
static const float SHARPEN = 0.88f;       // 88% CAS Sharpening (HD Clarity)

// =============================================================
// 2. SHADERS (Verified & Optimized)
// =============================================================

const char* vert = R"(#version 300 es
layout(location=0) in vec4 p; layout(location=1) in vec2 t; out mediump vec2 v;
void main(){gl_Position=p;v=t;})";

// --- PASS 1: VELOCITY ACCUMULATION ---
const char* frag_blur = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D c; // Current Frame
uniform sampler2D h; // History Frame
out vec4 o;

void main() {
    lowp vec4 curr = texture(c, v);
    lowp vec4 hist = texture(h, v);

    // 1. VELOCITY CALCULATOR (Anti-Ghosting)
    lowp float lC = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lH = dot(hist.rgb, vec3(0.299, 0.587, 0.114));
    lowp float diff = abs(lC - lH);

    // Dynamic Interpolation:
    // Low Diff (Walking) -> Max Blur (0.94)
    // High Diff (Flicking) -> Min Blur (0.35)
    lowp float velocity = smoothstep(0.02, 0.30, diff);
    lowp float factor = mix(0.94, 0.35, velocity);

    // 2. SHADOW PROTECTION (Contrast Fix)
    // If history is darker than current, we favor it slightly.
    // This prevents shadows from turning gray during movement.
    lowp vec4 result = mix(curr, hist, factor);
    if (lH < lC) { 
        result = mix(result, hist, 0.05); 
    }

    // 3. CENTER MASK (PvP Aim)
    // Protects the crosshair area (Radius 0.12)
    mediump vec2 center = vec2(0.5);
    lowp float dist = distance(v, center);
    lowp float mask = smoothstep(0.01, 0.12, dist);

    o = mix(curr, result, mask);
})";

// --- PASS 2: CLARITY & OUTPUT ---
const char* frag_draw = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D t;
out vec4 o;

void main() {
    // 1. CAS SHARPENING (Contrast Adaptive Sharpening)
    lowp vec4 col = texture(t, v);
    
    // Read 4 neighbors
    lowp vec4 n = textureOffset(t, v, ivec2(0, -1));
    lowp vec4 s = textureOffset(t, v, ivec2(0, 1));
    lowp vec4 e = textureOffset(t, v, ivec2(1, 0));
    lowp vec4 w = textureOffset(t, v, ivec2(-1, 0));

    // Calculate Luma for cheap/fast processing
    lowp float lC = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lN = dot(n.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lS = dot(s.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lE = dot(e.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lW = dot(w.rgb, vec3(0.299, 0.587, 0.114));

    // Calculate Contrast
    lowp float mx = max(lC, max(max(lN, lS), max(lE, lW)));
    lowp float mn = min(lC, min(min(lN, lS), min(lE, lW)));
    lowp float amt = sqrt(clamp(mn / (1.0 - mx + 0.001), 0.0, 1.0));
    
    // Apply Sharpening (Strength 0.88)
    lowp float peak = -1.0 / mix(8.0, 5.0, amt * 0.88); 
    lowp float sharpLuma = lC + (lN + lS + lE + lW) * peak;
    sharpLuma /= (1.0 + 4.0 * peak);
    
    // Apply Luma delta to Color
    col.rgb += (sharpLuma - lC);

    // 2. VIBRANCE (Color Restoration)
    // Boosts muted colors slightly to counter blur washout
    lowp float maxRGB = max(col.r, max(col.g, col.b));
    lowp float minRGB = min(col.r, min(col.g, col.b));
    lowp float sat = maxRGB - minRGB;
    col.rgb = mix(col.rgb, vec3(maxRGB), (1.0 - pow(sat, 0.5)) * -0.2);

    // 3. ACES TONEMAP
    lowp vec3 x = col.rgb;
    col.rgb = clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);

    // 4. ALPHA SAFETY (Fixes UI Bugs)
    o = vec4(col.rgb, 1.0);
})";

// =============================================================
// 3. RENDER ENGINE
// =============================================================
static GLuint rawTex=0, rawFBO=0, histTex[2]={0,0}, histFBO[2]={0,0}, vao=0;
static GLuint progBlur=0, progDraw=0;
static int ping=0, iW=0, iH=0, sW=0, sH=0;

void initGL(int w, int h) {
    // Resource cleanup
    if(rawTex){glDeleteTextures(1,&rawTex); glDeleteFramebuffers(1,&rawFBO); glDeleteTextures(2,histTex); glDeleteFramebuffers(2,histFBO); glDeleteVertexArrays(1,&vao);}
    
    // Internal Resolution
    iW=(int)(w*SCALE); iH=(int)(h*SCALE);

    // Shader Compilation
    auto c=[](GLenum t, const char* s){GLuint x=glCreateShader(t); glShaderSource(x,1,&s,0); glCompileShader(x); return x;};
    GLuint vs=c(GL_VERTEX_SHADER,vert), fs1=c(GL_FRAGMENT_SHADER,frag_blur), fs2=c(GL_FRAGMENT_SHADER,frag_draw);
    
    progBlur=glCreateProgram(); glAttachShader(progBlur,vs); glAttachShader(progBlur,fs1); glLinkProgram(progBlur);
    glUseProgram(progBlur); glUniform1i(glGetUniformLocation(progBlur,"c"),0); glUniform1i(glGetUniformLocation(progBlur,"h"),1);

    progDraw=glCreateProgram(); glAttachShader(progDraw,vs); glAttachShader(progDraw,fs2); glLinkProgram(progDraw);
    
    // Geometry Setup
    GLfloat d[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, 1,1,1,1}; GLushort i[]={0,1,2, 0,2,3};
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    GLuint vb,ib; glGenBuffers(1,&vb); glBindBuffer(GL_ARRAY_BUFFER,vb); glBufferData(GL_ARRAY_BUFFER,sizeof(d),d,GL_STATIC_DRAW);
    glGenBuffers(1,&ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(i),i,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,0,16,0); glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,0,16,(void*)8);

    // Texture Setup
    auto t = [&](GLuint& tx, GLuint& fb){
        glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_2D,tx);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,iW,iH,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tx,0);
    };
    t(rawTex,rawFBO); t(histTex[0],histFBO[0]); t(histTex[1],histFBO[1]);
    
    // Clear Buffers
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[0]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[1]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    sW=w; sH=h;
}

void render(int w, int h) {
    if(w!=sW || h!=sH || !rawTex) initGL(w,h);
    
    // Save state is not strictly required for SwapBuffers hooks on Android, 
    // but disabling tests is crucial for our full-screen pass.
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. FAST COPY (Downscale)
    glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,rawFBO);
    glBlitFramebuffer(0,0,w,h,0,0,iW,iH,GL_COLOR_BUFFER_BIT,GL_LINEAR);

    // 2. BLUR PASS
    int cur=ping, pre=1-ping;
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[cur]); glViewport(0,0,iW,iH); glBindVertexArray(vao);
    glUseProgram(progBlur);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,rawTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,histTex[pre]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    // 3. DRAW PASS (Upscale + Sharpen)
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,w,h);
    glUseProgram(progDraw);
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

__attribute__((constructor)) void init(){pthread_t t;pthread_create(&t,0,mainthread,0);}#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES3/gl3.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cmath>

#include "pl/Hook.h"
#include "pl/Gloss.h"

// =============================================================
// 1. FINAL SETTINGS
// =============================================================
static const float SCALE = 0.5f;          // 50% Internal Resolution (Max Performance)
static const float MAX_BLUR = 0.94f;      // 94% Smoothness (Walking/Looking around)
static const float MIN_BLUR = 0.35f;      // 35% Smoothness (Fast PvP Flicks)
static const float SHARPEN = 0.88f;       // 88% CAS Sharpening (HD Clarity)

// =============================================================
// 2. SHADERS (Verified & Optimized)
// =============================================================

const char* vert = R"(#version 300 es
layout(location=0) in vec4 p; layout(location=1) in vec2 t; out mediump vec2 v;
void main(){gl_Position=p;v=t;})";

// --- PASS 1: VELOCITY ACCUMULATION ---
const char* frag_blur = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D c; // Current Frame
uniform sampler2D h; // History Frame
out vec4 o;

void main() {
    lowp vec4 curr = texture(c, v);
    lowp vec4 hist = texture(h, v);

    // 1. VELOCITY CALCULATOR (Anti-Ghosting)
    lowp float lC = dot(curr.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lH = dot(hist.rgb, vec3(0.299, 0.587, 0.114));
    lowp float diff = abs(lC - lH);

    // Dynamic Interpolation:
    // Low Diff (Walking) -> Max Blur (0.94)
    // High Diff (Flicking) -> Min Blur (0.35)
    lowp float velocity = smoothstep(0.02, 0.30, diff);
    lowp float factor = mix(0.94, 0.35, velocity);

    // 2. SHADOW PROTECTION (Contrast Fix)
    // If history is darker than current, we favor it slightly.
    // This prevents shadows from turning gray during movement.
    lowp vec4 result = mix(curr, hist, factor);
    if (lH < lC) { 
        result = mix(result, hist, 0.05); 
    }

    // 3. CENTER MASK (PvP Aim)
    // Protects the crosshair area (Radius 0.12)
    mediump vec2 center = vec2(0.5);
    lowp float dist = distance(v, center);
    lowp float mask = smoothstep(0.01, 0.12, dist);

    o = mix(curr, result, mask);
})";

// --- PASS 2: CLARITY & OUTPUT ---
const char* frag_draw = R"(#version 300 es
precision mediump float;
in mediump vec2 v;
uniform sampler2D t;
out vec4 o;

void main() {
    // 1. CAS SHARPENING (Contrast Adaptive Sharpening)
    lowp vec4 col = texture(t, v);
    
    // Read 4 neighbors
    lowp vec4 n = textureOffset(t, v, ivec2(0, -1));
    lowp vec4 s = textureOffset(t, v, ivec2(0, 1));
    lowp vec4 e = textureOffset(t, v, ivec2(1, 0));
    lowp vec4 w = textureOffset(t, v, ivec2(-1, 0));

    // Calculate Luma for cheap/fast processing
    lowp float lC = dot(col.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lN = dot(n.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lS = dot(s.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lE = dot(e.rgb, vec3(0.299, 0.587, 0.114));
    lowp float lW = dot(w.rgb, vec3(0.299, 0.587, 0.114));

    // Calculate Contrast
    lowp float mx = max(lC, max(max(lN, lS), max(lE, lW)));
    lowp float mn = min(lC, min(min(lN, lS), min(lE, lW)));
    lowp float amt = sqrt(clamp(mn / (1.0 - mx + 0.001), 0.0, 1.0));
    
    // Apply Sharpening (Strength 0.88)
    lowp float peak = -1.0 / mix(8.0, 5.0, amt * 0.88); 
    lowp float sharpLuma = lC + (lN + lS + lE + lW) * peak;
    sharpLuma /= (1.0 + 4.0 * peak);
    
    // Apply Luma delta to Color
    col.rgb += (sharpLuma - lC);

    // 2. VIBRANCE (Color Restoration)
    // Boosts muted colors slightly to counter blur washout
    lowp float maxRGB = max(col.r, max(col.g, col.b));
    lowp float minRGB = min(col.r, min(col.g, col.b));
    lowp float sat = maxRGB - minRGB;
    col.rgb = mix(col.rgb, vec3(maxRGB), (1.0 - pow(sat, 0.5)) * -0.2);

    // 3. ACES TONEMAP
    lowp vec3 x = col.rgb;
    col.rgb = clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0);

    // 4. ALPHA SAFETY (Fixes UI Bugs)
    o = vec4(col.rgb, 1.0);
})";

// =============================================================
// 3. RENDER ENGINE
// =============================================================
static GLuint rawTex=0, rawFBO=0, histTex[2]={0,0}, histFBO[2]={0,0}, vao=0;
static GLuint progBlur=0, progDraw=0;
static int ping=0, iW=0, iH=0, sW=0, sH=0;

void initGL(int w, int h) {
    // Resource cleanup
    if(rawTex){glDeleteTextures(1,&rawTex); glDeleteFramebuffers(1,&rawFBO); glDeleteTextures(2,histTex); glDeleteFramebuffers(2,histFBO); glDeleteVertexArrays(1,&vao);}
    
    // Internal Resolution
    iW=(int)(w*SCALE); iH=(int)(h*SCALE);

    // Shader Compilation
    auto c=[](GLenum t, const char* s){GLuint x=glCreateShader(t); glShaderSource(x,1,&s,0); glCompileShader(x); return x;};
    GLuint vs=c(GL_VERTEX_SHADER,vert), fs1=c(GL_FRAGMENT_SHADER,frag_blur), fs2=c(GL_FRAGMENT_SHADER,frag_draw);
    
    progBlur=glCreateProgram(); glAttachShader(progBlur,vs); glAttachShader(progBlur,fs1); glLinkProgram(progBlur);
    glUseProgram(progBlur); glUniform1i(glGetUniformLocation(progBlur,"c"),0); glUniform1i(glGetUniformLocation(progBlur,"h"),1);

    progDraw=glCreateProgram(); glAttachShader(progDraw,vs); glAttachShader(progDraw,fs2); glLinkProgram(progDraw);
    
    // Geometry Setup
    GLfloat d[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, 1,1,1,1}; GLushort i[]={0,1,2, 0,2,3};
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    GLuint vb,ib; glGenBuffers(1,&vb); glBindBuffer(GL_ARRAY_BUFFER,vb); glBufferData(GL_ARRAY_BUFFER,sizeof(d),d,GL_STATIC_DRAW);
    glGenBuffers(1,&ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ib); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(i),i,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,0,16,0); glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,0,16,(void*)8);

    // Texture Setup
    auto t = [&](GLuint& tx, GLuint& fb){
        glGenTextures(1,&tx); glBindTexture(GL_TEXTURE_2D,tx);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,iW,iH,0,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tx,0);
    };
    t(rawTex,rawFBO); t(histTex[0],histFBO[0]); t(histTex[1],histFBO[1]);
    
    // Clear Buffers
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[0]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[1]); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    sW=w; sH=h;
}

void render(int w, int h) {
    if(w!=sW || h!=sH || !rawTex) initGL(w,h);
    
    // Save state is not strictly required for SwapBuffers hooks on Android, 
    // but disabling tests is crucial for our full-screen pass.
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. FAST COPY (Downscale)
    glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,rawFBO);
    glBlitFramebuffer(0,0,w,h,0,0,iW,iH,GL_COLOR_BUFFER_BIT,GL_LINEAR);

    // 2. BLUR PASS
    int cur=ping, pre=1-ping;
    glBindFramebuffer(GL_FRAMEBUFFER,histFBO[cur]); glViewport(0,0,iW,iH); glBindVertexArray(vao);
    glUseProgram(progBlur);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,rawTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,histTex[pre]);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);

    // 3. DRAW PASS (Upscale + Sharpen)
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,w,h);
    glUseProgram(progDraw);
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
