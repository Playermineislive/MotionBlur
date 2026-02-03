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
// 1. SETTINGS
// =============================================================
static const float BLUR_STRENGTH = 0.60f; // 0.60 is the sweet spot for PvP
static const float SCALE = 0.5f;          // Render at 50% resolution (Huge FPS Boost)

// =============================================================
// 2. ULTRA-FAST SHADERS (Low Precision)
// =============================================================

const char* vertexShaderSource = R"(#version 300 es
layout(location = 0) in vec4 aPosition;
layout(location = 1) in vec2 aTexCoord;
// OPTIMIZATION: Use mediump for coordinates (GLES2 trick)
out mediump vec2 vTexCoord; 
void main() {
    gl_Position = aPosition;
    vTexCoord = aTexCoord;
}
)";

const char* smartFragmentShaderSource = R"(#version 300 es
precision mediump float;
in mediump vec2 vTexCoord;
uniform sampler2D uCurrentFrame;
uniform sampler2D uHistoryFrame;
uniform lowp float uBlendFactor; // lowp is enough for simple 0-1 range
out vec4 FragColor;

void main() {
    // OPTIMIZATION: Use lowp for colors (Faster on Mali/Adreno GPUs)
    lowp vec4 current = texture(uCurrentFrame, vTexCoord);
    lowp vec4 history = texture(uHistoryFrame, vTexCoord);

    // Fast difference check (Manhattan Distance - No Sqrt)
    lowp vec3 diffVec = abs(current.rgb - history.rgb);
    lowp float diff = diffVec.r + diffVec.g + diffVec.b;

    lowp float smartFactor = uBlendFactor;
    
    // SMART LOGIC: If difference > 0.4 (Menu Opened), drop blur to 0.
    if (diff > 0.4) { 
        smartFactor = 0.0;
    }

    FragColor = mix(current, history, smartFactor); 
}
)";

const char* drawFragmentShaderSource = R"(#version 300 es
precision mediump float;
in mediump vec2 vTexCoord;
uniform sampler2D uTexture;
out vec4 FragColor;
void main() {
    FragColor = texture(uTexture, vTexCoord);
}
)";

// =============================================================
// 3. GLOBAL VARIABLES
// =============================================================

static GLuint rawTexture = 0, rawFBO = 0;
static GLuint historyTextures[2] = {0, 0};
static GLuint historyFBOs[2] = {0, 0};
static GLuint vao = 0, vbo = 0, ibo = 0;

static GLuint blendProgram = 0, drawProgram = 0;
static GLint locCurrent = -1, locHistory = -1, locFactor = -1, locTexture = -1;

static int pingPongIndex = 0;
static bool isFirstFrame = true;
static int savedWidth = 0, savedHeight = 0;
static int internalW = 0, internalH = 0;

// =============================================================
// 4. RESOURCE INITIALIZATION
// =============================================================

void initResources(int width, int height) {
    if (rawTexture != 0) {
        glDeleteTextures(1, &rawTexture); glDeleteFramebuffers(1, &rawFBO);
        glDeleteTextures(2, historyTextures); glDeleteFramebuffers(2, historyFBOs);
        glDeleteVertexArrays(1, &vao);
    }

    // CALCULATE SCALED RESOLUTION
    internalW = (int)(width * SCALE);
    internalH = (int)(height * SCALE);

    auto compile = [](GLenum type, const char* src) {
        GLuint s = glCreateShader(type); glShaderSource(s, 1, &src, 0); glCompileShader(s); return s;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fsBlend = compile(GL_FRAGMENT_SHADER, smartFragmentShaderSource);
    GLuint fsDraw = compile(GL_FRAGMENT_SHADER, drawFragmentShaderSource);

    blendProgram = glCreateProgram();
    glAttachShader(blendProgram, vs); glAttachShader(blendProgram, fsBlend); glLinkProgram(blendProgram);
    locCurrent = glGetUniformLocation(blendProgram, "uCurrentFrame");
    locHistory = glGetUniformLocation(blendProgram, "uHistoryFrame");
    locFactor = glGetUniformLocation(blendProgram, "uBlendFactor");

    drawProgram = glCreateProgram();
    glAttachShader(drawProgram, vs); glAttachShader(drawProgram, fsDraw); glLinkProgram(drawProgram);
    locTexture = glGetUniformLocation(drawProgram, "uTexture");

    // Standard Geometry (Full Screen Quad)
    GLfloat verts[] = { -1,1,0,1,  -1,-1,0,0,  1,-1,1,0,  1,1,1,1 };
    GLushort inds[] = { 0,1,2, 0,2,3 };
    
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(inds), inds, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, 0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
    glBindVertexArray(0);

    // TEXTURE SETUP (Scaled Down)
    auto setupTex = [&](GLuint& tex, GLuint& fbo) {
        glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
        // CRITICAL: Use internalW/internalH (Smaller)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, internalW, internalH, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    };

    setupTex(rawTexture, rawFBO);
    for(int i=0; i<2; i++) {
        setupTex(historyTextures[i], historyFBOs[i]);
        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT); // Clear Alpha Channel
    }

    savedWidth = width; savedHeight = height;
    isFirstFrame = true;
}

// =============================================================
// 5. RENDER PIPELINE
// =============================================================

void render_effect(int width, int height) {
    if (width != savedWidth || height != savedHeight || rawTexture == 0) initResources(width, height);

    GLint lastFBO; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &lastFBO);
    glDisable(GL_SCISSOR_TEST); glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);

    // 1. DOWNSCALE & COPY (Fast Blit)
    // Screen (High Res) -> Texture (Low Res)
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rawFBO);
    glBlitFramebuffer(0, 0, width, height, 0, 0, internalW, internalH, GL_COLOR_BUFFER_BIT, GL_LINEAR);

    // 2. APPLY BLUR (At Low Res)
    int curr = pingPongIndex;
    int prev = 1 - pingPongIndex;
    glBindVertexArray(vao);
    glViewport(0, 0, internalW, internalH); // Render at small scale

    if (isFirstFrame) {
        glBindFramebuffer(GL_FRAMEBUFFER, historyFBOs[curr]);
        glUseProgram(drawProgram);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, rawTexture);
        glUniform1i(locTexture, 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        isFirstFrame = false;
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, historyFBOs[curr]);
        glUseProgram(blendProgram);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, rawTexture);
        glUniform1i(locCurrent, 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, historyTextures[prev]);
        glUniform1i(locHistory, 1);
        glUniform1f(locFactor, BLUR_STRENGTH);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
    }

    // 3. UPSCALE & DRAW TO SCREEN
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height); // Restore full scale
    
    glUseProgram(drawProgram);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, historyTextures[curr]);
    glUniform1i(locTexture, 0);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    glBindVertexArray(0);
    pingPongIndex = prev;
}

// =============================================================
// 6. HOOK
// =============================================================

static EGLBoolean (*orig_swap)(EGLDisplay, EGLSurface) = nullptr;

static EGLBoolean hook_swap(EGLDisplay dpy, EGLSurface surf) {
    if (orig_swap) {
        EGLint w, h;
        eglQuerySurface(dpy, surf, EGL_WIDTH, &w);
        eglQuerySurface(dpy, surf, EGL_HEIGHT, &h);
        
        // Only run if the surface is valid game size
        if (w > 100) render_effect(w, h);
        
        return orig_swap(dpy, surf);
    }
    return EGL_FALSE;
}

static void* mainthread(void*) {
    sleep(1);
    GlossInit(true);
    GHandle hegl = GlossOpen("libEGL.so");
    void* swap = (void*)GlossSymbol(hegl, "eglSwapBuffers", nullptr);
    if(swap) GlossHook(swap, (void*)hook_swap, (void**)&orig_swap);
    return nullptr;
}

__attribute__((constructor))
void lib_init() {
    pthread_t t;
    pthread_create(&t, nullptr, mainthread, nullptr);
}
