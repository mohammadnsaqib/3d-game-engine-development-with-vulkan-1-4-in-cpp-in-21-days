// =======================================================================================
// MERIDIAN - Day 6.  A matrix, a push constant, and the first object with an inside.
//
// FIVE DAYS OF FLAT PICTURES END HERE. Every vertex so far has been a pair of numbers already
// in clip space, which meant the engine could not move anything, turn anything, or look at
// anything from somewhere else. Today a vertex has three coordinates and a 4x4 matrix decides
// where it lands.
//
// THREE MATRICES, MULTIPLIED IN ONE ORDER:
//
//   model       where the object is, and which way round -- one per object
//   view        where the camera is. It is the INVERSE of the camera's own placement, which
//               is the single most confusing sentence in 3D graphics and Day 13 earns it.
//   projection  what lens the camera has: field of view, aspect, near and far -- and the flip
//
// AND THE FLIP FINALLY LIVES SOMEWHERE. Vulkan's y points down. Day 1 photographed the
// upside-down triangle, Day 3 declined to fix it in the blit, Day 5 found it had reversed a
// winding. Today it is ONE NEGATED NUMBER in the projection matrix, and every one of those
// three problems stops existing.
//
// A PUSH CONSTANT CARRIES THE RESULT. Not a buffer, not a descriptor -- 64 bytes recorded into
// the command buffer beside the draw. Day 9 is about what to do when 128 bytes is not enough.
//
// AND THERE IS NO DEPTH BUFFER YET, which is deliberate. A convex solid with backface culling
// on needs no depth test at all: the faces you can see are exactly the ones facing you. The
// last scene draws twenty-seven cubes to show what that stops being true of, and Day 7 fixes
// it in one attachment.
//
//   meridian.exe               renders every scene below and exits
//   meridian.exe --all         the same thing, said out loud
//   meridian.exe --scene NAME  renders one of them
//   meridian.exe --window      open a window and present until Esc
//   meridian.exe --provoke     break a rule on purpose, and print what validation says
// =======================================================================================
#define WIN32_LEAN_AND_MEAN
// VK_USE_PLATFORM_WIN32_KHR IS WHAT MAKES VkWin32SurfaceCreateInfoKHR EXIST. The Vulkan header
// declares every platform's surface types behind a define, because a header that declared all
// of them would need every platform s own headers to be present. This is the one line in the
// book that says which operating system it is.
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <wincodec.h>
#include <vulkan/vulkan.h>

#include "vk_dispatch.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "spirv.h"

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

// THE CLEAR COLOUR, and it is the same night blue the whole book is lit against.
static const float NIGHT[4] = { 0.055f, 0.066f, 0.098f, 1.0f };

static const uint32_t RENDER_W = 1920, RENDER_H = 1080;

// ---- THE GEOMETRY ---------------------------------------------------------------------
// A VERTEX HAS THREE COORDINATES NOW, and that one change is what makes everything below
// necessary. Twenty-four bytes: a position in the object's own space and a colour.
struct Vertex {
    float x, y, z;     // location 0, VK_FORMAT_R32G32B32_SFLOAT, offset 0
    float r, g, b;     // location 1, VK_FORMAT_R32G32B32_SFLOAT, offset 12
};

// ---- FOUR BY FOUR, COLUMN MAJOR, AND WRITTEN OUT RATHER THAN LINKED IN --------------------
// GLM would do all of this in a header. It is fifty lines here because a reader who has never
// multiplied two matrices by hand should see what the multiplication IS before importing one.
//
// COLUMN MAJOR because that is what GLSL expects: m[column][row], and element c*4+r. Getting
// this backwards transposes every matrix in the program and produces a picture that is almost
// right, which is far worse than one that is obviously wrong.
struct Mat4 { float m[16]; };

static Mat4 identity() {
    Mat4 r = {};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

// a * b, meaning "apply b, then a" -- the order everybody gets wrong once.
static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r = {};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
    return r;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x; r.m[13] = y; r.m[14] = z;
    return r;
}

static Mat4 rotate_y(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
    return r;
}

static Mat4 rotate_x(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
    return r;
}

// THE VIEW MATRIX IS THE CAMERA'S INVERSE. A camera at (0, 0, 4) looking at the origin does not
// move the camera -- it moves the WORLD four units towards the viewer. Everything here is
// built from that one idea.
static Mat4 look_at(float ex, float ey, float ez) {
    float fx = -ex, fy = -ey, fz = -ez;                    // forward: eye towards origin
    const float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0.0f - fz * 1.0f, sy = fz * 0.0f - fx * 0.0f, sz = fx * 1.0f - fy * 0.0f;
    const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
    sx /= sl; sy /= sl; sz /= sl;
    const float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    Mat4 r = identity();
    r.m[0] = sx; r.m[4] = sy; r.m[8]  = sz;
    r.m[1] = ux; r.m[5] = uy; r.m[9]  = uz;
    r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
    r.m[12] = -(sx * ex + sy * ey + sz * ez);
    r.m[13] = -(ux * ex + uy * ey + uz * ez);
    r.m[14] = fx * ex + fy * ey + fz * ez;
    return r;
}

// THE PROJECTION, AND THE ONE NEGATED NUMBER THIS BOOK HAS BEEN WAITING FOR.
//
//   m[5] = -cot(fov/2)   the y flip, because Vulkan's y points DOWN
//   depth 0..1, not -1..1 -- OpenGL's range is the odd one out, not Vulkan's
static Mat4 perspective(float fov_y, float aspect, float znear, float zfar, bool flip) {
    const float f = 1.0f / std::tan(fov_y * 0.5f);
    Mat4 r = {};
    r.m[0] = f / aspect;
    r.m[5] = flip ? -f : f;
    r.m[10] = zfar / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (znear * zfar) / (znear - zfar);
    return r;
}

// NO PERSPECTIVE DIVIDE AT ALL: parallel lines stay parallel, and distance stops mattering.
static Mat4 ortho(float half, float aspect, float znear, float zfar, bool flip) {
    Mat4 r = identity();
    r.m[0] = 1.0f / (half * aspect);
    r.m[5] = (flip ? -1.0f : 1.0f) / half;
    r.m[10] = 1.0f / (znear - zfar);
    r.m[14] = znear / (znear - zfar);
    return r;
}

// A CUBE WITH TWENTY-FOUR VERTICES, not eight. Eight would share every corner between three
// faces, and a shared corner cannot have three different colours -- which is the same reason
// Day 11's normals will force the count up again. A vertex is not a point in space; it is
// every attribute a face needs at that point.
static void build_cube(std::vector<Vertex>& v, std::vector<uint32_t>& idx) {
    struct Face { float nx, ny, nz; float r, g, b; };
    static const Face F[6] = {
        {  0,  0,  1, 1.00f, 0.36f, 0.18f },   // +z  ember
        {  0,  0, -1, 0.20f, 0.55f, 1.00f },   // -z  sky
        {  1,  0,  0, 0.98f, 0.78f, 0.25f },   // +x  brass
        { -1,  0,  0, 0.25f, 0.85f, 0.52f },   // -x  moss
        {  0,  1,  0, 0.85f, 0.30f, 0.75f },   // +y  orchid
        {  0, -1,  0, 0.55f, 0.58f, 0.68f },   // -y  stone
    };
    for (int f = 0; f < 6; ++f) {
        const Face& fa = F[f];
        // two vectors across the face, from its normal
        float ax = fa.ny, ay = fa.nz, az = fa.nx;
        const float bx = fa.ny * az - fa.nz * ay;
        const float by = fa.nz * ax - fa.nx * az;
        const float bz = fa.nx * ay - fa.ny * ax;
        const uint32_t base = uint32_t(v.size());
        const float sx[4] = { -1, 1, 1, -1 }, sy[4] = { -1, -1, 1, 1 };
        for (int i = 0; i < 4; ++i)
            v.push_back({ fa.nx + ax * sx[i] + bx * sy[i],
                          fa.ny + ay * sx[i] + by * sy[i],
                          fa.nz + az * sx[i] + bz * sy[i],
                          fa.r, fa.g, fa.b });
        const uint32_t q[6] = { base, base + 1, base + 2, base, base + 2, base + 3 };
        idx.insert(idx.end(), q, q + 6);
    }
}

// ---- THE SCENES -----------------------------------------------------------------------
// One cube, one pipeline layout, and ten answers to "where is the camera and what lens has
// it". Every difference below ends up in sixteen floats.
enum Proj { PROJ_PERSPECTIVE, PROJ_ORTHO };

struct Scene {
    const char* name;
    Proj        proj;
    float       fov_deg;      // or, for ortho, the half-height it frames
    float       distance;     // how far back the camera sits
    float       yaw, pitch;   // where the cube is turned to
    bool        flip;         // the negated m[5]
    bool        cull;
    int         grid;         // 1 cube, or a 3x3x3 lattice of them
    const char* note;
};

static const Scene SCENES[] = {
    { "cube",     PROJ_PERSPECTIVE, 45.0f, 4.6f, 0.62f,  0.42f, true,  true,  1,
      "one cube, three faces, and a matrix" },
    { "spin",     PROJ_PERSPECTIVE, 45.0f, 4.6f, 2.30f, -0.55f, true,  true,  1,
      "the same cube, turned somewhere else" },
    { "wide",     PROJ_PERSPECTIVE, 90.0f, 2.6f, 0.62f,  0.42f, true,  true,  1,
      "a 90 degree lens, close in" },
    { "narrow",   PROJ_PERSPECTIVE, 18.0f, 11.0f, 0.62f, 0.42f, true,  true,  1,
      "an 18 degree lens, far back: flat" },
    { "ortho",    PROJ_ORTHO,        2.2f, 4.6f, 0.62f,  0.42f, true,  true,  1,
      "no perspective divide at all" },
    { "noflip",   PROJ_PERSPECTIVE, 45.0f, 4.6f, 0.62f,  0.42f, false, true,  1,
      "m[5] positive: the answer for five days" },
    { "nocull",   PROJ_PERSPECTIVE, 45.0f, 4.6f, 0.62f,  0.42f, true,  false, 1,
      "cullMode NONE: the inside shows through" },
    { "far",      PROJ_PERSPECTIVE, 45.0f, 13.0f, 0.62f, 0.42f, true,  true,  1,
      "the same lens, the camera pulled back" },
    { "close",    PROJ_PERSPECTIVE, 45.0f, 2.1f, 0.62f,  0.42f, true,  true,  1,
      "the camera almost touching it" },
    { "grid",     PROJ_PERSPECTIVE, 45.0f, 13.5f, 0.62f, 0.42f, true, true,  3,
      "27 cubes, 27 push constants, and NO depth buffer" },
};
static const int SCENE_COUNT = int(sizeof SCENES / sizeof SCENES[0]);

struct Rect { int32_t x0, y0, x1, y1; };
enum Fit { FIT_FILL, FIT_LETTERBOX, FIT_STRETCH };

// WHERE A SOURCE OF ONE SHAPE LANDS INSIDE A DESTINATION OF ANOTHER -- Day 3's three answers,
// kept because the window still needs them. Today every scene is the same shape as its
// destination, so this returns the whole rectangle every time.
static Rect fit_into(Fit how, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    if (how == FIT_STRETCH || how == FIT_FILL)
        return { 0, 0, int32_t(dw), int32_t(dh) };
    const double sa = double(sw) / double(sh), da = double(dw) / double(dh);
    if (da > sa) {
        const int32_t w = int32_t(double(dh) * sa + 0.5);
        return { int32_t((dw - w) / 2), 0, int32_t((dw - w) / 2) + w, int32_t(dh) };
    }
    const int32_t h = int32_t(double(dw) / sa + 0.5);
    return { 0, int32_t((dh - h) / 2), int32_t(dw), int32_t((dh - h) / 2) + h };
}

// =======================================================================================
// VALIDATION.  The most important twenty lines in the book.
// =======================================================================================
// Vulkan does almost no checking of its own: a wrong enum, a barrier you forgot, an image in
// the wrong layout -- the driver is entitled to do whatever is fastest, which is usually to
// produce a picture that is wrong, or nothing, and say nothing at all.
//
// THE VALIDATION LAYER IS A SEPARATE PIECE OF SOFTWARE that sits between your program and the
// driver and checks every call against the specification. It costs perhaps a third of the
// frame and it is worth every microsecond until Day 21. Leave it on.
static int g_messages = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    ++g_messages;
    const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR"
                                                                                  : "warn";
    // WRAPPED AT 74 COLUMNS, because a validation message is a paragraph, not a word, and the
    // half that gets cut is always the half that names the rule you broke.
    //
    // AND SPLIT ON ITS OWN NEWLINES FIRST. The message already contains line breaks -- the
    // rule, then a break, then "The Vulkan spec states:" -- and wrapping on width alone leaves
    // one of them in the middle of a wrapped line with no indent under it.
    const std::string body = d->pMessage ? d->pMessage : "";
    std::string lead = std::string("  vk ") + level + "  ";
    size_t start = 0;
    while (start <= body.size()) {
        const size_t nl = body.find('\n', start);
        std::string line = lead + body.substr(start, nl == std::string::npos
                                                     ? std::string::npos : nl - start);
        const size_t WRAP = 74;
        while (line.size() > WRAP) {
            size_t cut = line.rfind(' ', WRAP);
            if (cut == std::string::npos || cut < 12) cut = WRAP;
            std::printf("%s\n", line.substr(0, cut).c_str());
            line = "           " + line.substr(cut + 1);
        }
        std::printf("%s\n", line.c_str());
        if (nl == std::string::npos) break;
        start = nl + 1;
        lead = "           ";
    }
    return VK_FALSE;
}

// A result that is not VK_SUCCESS is a bug, and stopping on it is better than carrying on with
// a handle that was never created.
#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) {                  \
    std::printf("FAILED: %s = %d (line %d)\n", #x, (int)_r, __LINE__);               \
    std::exit(2); } } while (0)

// =======================================================================================
// THE PICTURE COMES HOME.  WIC, and the one thing it does that nobody expects.
// =======================================================================================
static bool write_png(const wchar_t* path, const uint8_t* rgba, UINT w, UINT h, UINT stride,
                      bool* swapped) {
    IWICImagingFactory* factory = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&factory)))) return false;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IWICStream* stream = nullptr;
    bool ok = false;
    if (SUCCEEDED(factory->CreateStream(&stream)) &&
        SUCCEEDED(stream->InitializeFromFilename(path, GENERIC_WRITE)) &&
        SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) &&
        SUCCEEDED(enc->Initialize(stream, WICBitmapEncoderNoCache)) &&
        SUCCEEDED(enc->CreateNewFrame(&frame, nullptr)) &&
        SUCCEEDED(frame->Initialize(nullptr)) &&
        SUCCEEDED(frame->SetSize(w, h))) {
        // ASK FOR RGBA AND READ BACK WHAT YOU WERE GIVEN. SetPixelFormat takes the format you
        // want, answers with the one you are getting, and returns S_OK either way. On this
        // machine it answers BGRA every time, so the swap is MEASURED rather than assumed.
        WICPixelFormatGUID want = GUID_WICPixelFormat32bppRGBA, got = want;
        frame->SetPixelFormat(&got);
        *swapped = (got != want);
        std::vector<uint8_t> buf(size_t(w) * h * 4);
        for (UINT y = 0; y < h; ++y)
            std::memcpy(&buf[size_t(y) * w * 4], rgba + size_t(y) * stride, size_t(w) * 4);
        if (*swapped)
            for (size_t i = 0; i < buf.size(); i += 4) std::swap(buf[i], buf[i + 2]);
        if (SUCCEEDED(frame->WritePixels(h, w * 4, (UINT)buf.size(), buf.data())) &&
            SUCCEEDED(frame->Commit()) && SUCCEEDED(enc->Commit()))
            ok = true;
    }
    if (frame) frame->Release();
    if (enc) enc->Release();
    if (stream) stream->Release();
    factory->Release();
    return ok;
}

// =======================================================================================
// MEMORY.  Vulkan has no heaps with names; it has a table, and you choose.
// =======================================================================================
// Direct3D 12 has an UPLOAD heap and a DEFAULT heap. OpenGL has usage hints. Vulkan has a list
// of memory TYPES, each with property flags and a heap it comes from, and the question is
// always the same: which types will this resource accept, and which of those has the
// properties I need? The first that satisfies both is the answer.
//
// THE ALLOWED MASK COMES FROM THE RESOURCE, NOT FROM YOU. vkGetImageMemoryRequirements fills
// in memoryTypeBits, and a type not in that mask cannot hold that image no matter how well its
// flags read.
static uint32_t memory_type(VkPhysicalDevice pd, uint32_t allowed, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties m = {};
    vkGetPhysicalDeviceMemoryProperties(pd, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
        if ((allowed & (1u << i)) && (m.memoryTypes[i].propertyFlags & want) == want)
            return i;
    std::printf("FAILED: no memory type for mask 0x%X with flags 0x%X\n", allowed, want);
    std::exit(3);
}

// A BUFFER IS A SIZE, A USAGE, AND -- SEPARATELY -- SOME MEMORY. vkCreateBuffer allocates
// nothing. It makes an object that knows how big it is and what it is for, and that object
// then TELLS YOU which memory types it will accept. Binding memory to it is a second call, and
// forgetting it is the most common first-week Vulkan bug there is.
struct Buffer {
    VkBuffer       handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    uint32_t       type_index = ~0u;
    bool           host_visible = false;
};

// Everything the engine keeps for its whole run. Day 7 gives this a name and a lifetime; today
// it is a struct and a function that fills it.
struct Context {
    VkInstance       instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice         device = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         family = ~0u;
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkShaderModule   vs_mesh = VK_NULL_HANDLE;
    VkShaderModule   fs = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

    // TWO BUFFERS. Yesterday's nine were a tour of what a buffer can be; from today the
    // engine keeps what it draws, and it draws a cube.
    Buffer cube_vb, cube_ib;
    uint32_t cube_indices = 0;
    uint32_t max_push = 0;
    double staged_us = 0.0;
    // THE CACHE IS AN OBJECT, not a setting. It is handed to every vkCreateGraphicsPipelines
    // call, it fills itself, and vkGetPipelineCacheData hands the bytes back for you to write
    // to disk -- which is how a shipped game skips this cost on the second launch.
    VkPipelineCache cache = VK_NULL_HANDLE;
};

static VkShaderModule make_module(VkDevice dev, const uint32_t* words, size_t bytes) {
    VkShaderModuleCreateInfo smi = {};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = bytes;
    smi.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &m));
    return m;
}

static void make_context(Context& c, bool validation) {
    // ---- 1. WHAT THE LOADER OFFERS ------------------------------------------------------
    uint32_t api = 0;
    VK_CHECK(vkEnumerateInstanceVersion(&api));
    std::printf("loader   : instance version %u.%u.%u\n", VK_API_VERSION_MAJOR(api),
                VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));

    uint32_t nlayers = 0;
    vkEnumerateInstanceLayerProperties(&nlayers, nullptr);
    std::vector<VkLayerProperties> layers(nlayers);
    vkEnumerateInstanceLayerProperties(&nlayers, layers.data());
    bool have = false;
    for (const VkLayerProperties& l : layers)
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) have = true;
    std::printf("layers   : %u available, validation is %s\n", nlayers,
                have ? "PRESENT" : "ABSENT");
    validation = validation && have;

    // ---- 2. THE INSTANCE -----------------------------------------------------------------
    // THE MESSENGER IS CHAINED INTO pNext, not created afterwards. A messenger created after
    // the instance cannot report anything that went wrong DURING vkCreateInstance, which is
    // exactly when a mistyped layer name or a missing extension shows up.
    VkDebugUtilsMessengerCreateInfoEXT dbg = {};
    dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg.pfnUserCallback = on_message;

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Meridian";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Meridian";
    // WHAT apiVersion ACTUALLY DOES: it is the highest version whose behaviour you promise to
    // understand. Ask for 1.4 and you get 1.4's rules; ask for 1.0 on a 1.4 driver and you get
    // 1.0's, including the ones that were later relaxed.
    app.apiVersion = VK_API_VERSION_1_4;

    const char* want_layers[] = { "VK_LAYER_KHRONOS_validation" };
    // PRESENTATION IS AN EXTENSION, at the instance level as well as the device level. Vulkan
    // runs on machines with no screen, so a surface is something you ask for.
    const char* want_exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                                VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_KHR_WIN32_SURFACE_EXTENSION_NAME };

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = validation ? &dbg : nullptr;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = validation ? 1u : 0u;
    ici.ppEnabledLayerNames = want_layers;
    // The surface extensions are always needed; the debug one only with validation on, and
    // it is first in the array so the count can simply drop by one.
    ici.enabledExtensionCount = validation ? 3u : 2u;
    ici.ppEnabledExtensionNames = validation ? want_exts : want_exts + 1;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.instance));

    // THE MESSENGER'S OWN FUNCTION IS AN EXTENSION, so it is not in vulkan-1.lib and has to be
    // fetched. Day 2 is about why every Vulkan function works this way underneath.
    if (validation) {
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(c.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create) VK_CHECK(create(c.instance, &dbg, nullptr, &c.messenger));
    }
    std::printf("instance : created, asking for Vulkan 1.4%s\n",
                validation ? ", validation on" : ", validation OFF");

    // ---- 3. THE PHYSICAL DEVICE ----------------------------------------------------------
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(c.instance, &ndev, nullptr);
    if (!ndev) { std::printf("FAILED: no Vulkan device\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(c.instance, &ndev, devs.data());
    std::printf("devices  : %u\n", ndev);

    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties p = {};
        vkGetPhysicalDeviceProperties(d, &p);
        const char* kind = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
                         : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                         : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "software" : "other";
        std::printf("  %-34s %-10s api %u.%u.%u\n", p.deviceName, kind,
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion));

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i) {
            if (!(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            // A DISCRETE CARD WINS. On a laptop the integrated chip is usually first in the
            // list and will happily draw this at a fifth of the speed, which is the single
            // most common "why is it slow" in Windows graphics.
            if (c.gpu == VK_NULL_HANDLE ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                c.gpu = d;
                c.family = i;
            }
            break;
        }
    }
    if (c.gpu == VK_NULL_HANDLE) { std::printf("FAILED: no graphics queue\n"); std::exit(1); }

    VkPhysicalDeviceProperties gp = {};
    vkGetPhysicalDeviceProperties(c.gpu, &gp);
    std::printf("gpu      : %s\n", gp.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &nq, qs.data());
    std::printf("queues   : %u family(ies) on this card\n", nq);
    for (uint32_t i = 0; i < nq; ++i)
        std::printf("  family %u  %2u queue(s)  %s%s%s%s\n", i, qs[i].queueCount,
                    (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "graphics " : "",
                    (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "compute " : "",
                    (qs[i].queueFlags & VK_QUEUE_TRANSFER_BIT) ? "transfer " : "",
                    (qs[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "sparse" : "");

    // ---- 4. THE LOGICAL DEVICE ------------------------------------------------------------
    // A CORE FEATURE IS STILL OFF UNTIL YOU ASK. dynamicRendering and synchronization2 have
    // been core since 1.3 and a 1.4 device still starts with them disabled. Using one without
    // enabling it is a validation error, not a crash, which is the whole argument for the
    // layer.
    VkPhysicalDeviceVulkan13Features f13 = {};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    // fillModeNonSolid is what lets a pipeline ask for lines or points instead of filled
    // triangles. It is a device feature rather than a state, and two of today's scenes need it.
    VkPhysicalDeviceFeatures base = {};
    base.fillModeNonSolid = VK_TRUE;
    // wideLines lets a pipeline set lineWidth to anything but 1.0. Like fillModeNonSolid it
    // is OPTIONAL: a card is allowed to say no, and asking for a feature the card does not
    // have makes vkCreateDevice fail rather than quietly ignore you.
    base.wideLines = VK_TRUE;
    // largePoints is what lets gl_PointSize be anything but 1.0, and like the other two it is
    // OPTIONAL. One of today's scenes draws the ring's vertices as nine-pixel points.
    base.largePoints = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2 = {};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f13;
    f2.features = base;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    VK_CHECK(vkCreateDevice(c.gpu, &dci, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.family, 0, &c.queue);
    std::printf("device   : queue family %u, dynamic rendering and sync2 enabled\n", c.family);

    // ---- THE DISPATCH TABLE, AND EVERY CALL BELOW THIS LINE GOES STRAIGHT TO THE DRIVER ----
    // Until this call, every vkCmd* in this program was the loader's trampoline. After it, the
    // macros in vk_dispatch.h have redirected them to function pointers the driver handed over
    // -- and not one call site changed.
    const int missing = load_device_dispatch(c.device, vk);
    std::printf("dispatch : %d device-level entry point(s) from the registry, %d missing\n",
                DEVICE_DISPATCH_COUNT, missing);
    if (missing) std::exit(1);

    // ---- 5. THE PIECES EVERY SCENE SHARES --------------------------------------------------
    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = c.family;
    VK_CHECK(vkCreateCommandPool(c.device, &cpi, nullptr, &c.pool));

    c.vs_mesh = make_module(c.device, kMeshVert, sizeof kMeshVert);
    c.fs = make_module(c.device, kTriFrag, sizeof kTriFrag);
    std::printf("spirv    : %zu and %zu words, compiled at BUILD time\n",
                sizeof kMeshVert / 4, sizeof kTriFrag / 4);

    // ---- THE LAYOUT STOPS BEING EMPTY ------------------------------------------------------
    // A PUSH CONSTANT RANGE IS AN OFFSET, A SIZE AND WHICH STAGES CAN SEE IT. Sixty-four bytes
    // for one mat4, visible to the vertex stage. The guaranteed minimum across every Vulkan
    // implementation is 128 bytes -- two matrices -- and a range larger than the device's own
    // maxPushConstantsSize fails at layout creation, not at draw time.
    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    range.size = sizeof(Mat4);

    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.layout));

    VkPhysicalDeviceProperties limits = {};
    vkGetPhysicalDeviceProperties(c.gpu, &limits);
    c.max_push = limits.limits.maxPushConstantsSize;
    std::printf("push     : %u bytes of range, %u available, 128 guaranteed anywhere\n",
                uint32_t(sizeof(Mat4)), c.max_push);
}

// =======================================================================================
// BUFFERS, AND THE TWO WAYS BYTES GET INTO ONE.
// =======================================================================================
static void make_buffer(Context& c, Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                        bool host_visible) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device, &bci, nullptr, &b.handle));

    // THE ALLOWED MASK COMES FROM THE BUFFER. memoryTypeBits is the driver saying which of the
    // card's memory types this particular buffer, with this particular usage, can live in.
    VkMemoryRequirements mr = {};
    vkGetBufferMemoryRequirements(c.device, b.handle, &mr);
    const VkMemoryPropertyFlags want =
        host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // mr.size IS NOT size. The driver may want more, for alignment or for its own page
    // granularity, and allocating what you asked for rather than what it asked for is a
    // validation error on some cards and a silent corruption on others.
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits, want);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(c.device, b.handle, b.memory, 0));

    b.size = mr.size;
    b.type_index = mai.memoryTypeIndex;
    b.host_visible = host_visible;
}

static void destroy_buffer(Context& c, Buffer& b) {
    if (b.handle) vkDestroyBuffer(c.device, b.handle, nullptr);
    if (b.memory) vkFreeMemory(c.device, b.memory, nullptr);
    b.handle = VK_NULL_HANDLE;
    b.memory = VK_NULL_HANDLE;
}

// PATH ONE: MAP IT AND WRITE. Only possible on HOST_VISIBLE memory, which on a discrete card is
// across the PCI Express bus from the shader cores. Simple, immediate, and the memory the
// vertex fetch reads from is the slow one.
static void write_mapped(Context& c, Buffer& b, const void* data, size_t bytes) {
    void* dst = nullptr;
    VK_CHECK(vkMapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &dst));
    std::memcpy(dst, data, bytes);
    // HOST_COHERENT is why there is no flush here. Without it the write is not guaranteed
    // visible to the card until vkFlushMappedMemoryRanges, and leaving that out is a bug that
    // reproduces on one vendor in ten.
    vkUnmapMemory(c.device, b.memory);
}

// PATH TWO: STAGE IT. Write into a throwaway HOST_VISIBLE buffer, record a copy into the
// DEVICE_LOCAL one, submit, wait, destroy the staging buffer. Three times the code, and the
// memory the card reads from every frame afterwards is its own.
static void upload_staged(Context& c, Buffer& dst, const void* data, size_t bytes) {
    Buffer staging;
    make_buffer(c, staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    write_mapped(c, staging, data, bytes);

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    VkBufferCopy region = {};
    region.size = bytes;
    vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, fence));
    // THE WAIT IS WHY THIS BELONGS AT LOAD TIME. A staged upload in the middle of a frame that
    // waits like this has stopped the CPU dead; Day 18 does it without waiting.
    VK_CHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    destroy_buffer(c, staging);
}

// A CLOCK, because "staging is faster" is a claim and this book does not print claims.
static double now_us() {
    LARGE_INTEGER f = {}, t = {};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return double(t.QuadPart) * 1e6 / double(f.QuadPart);
}

// EVERY BUFFER THE ENGINE OWNS, BUILT ONCE. The ring vertex buffer is deliberately allocated
// larger than its contents so the wrong-stride scene reads garbage that is still INSIDE the
// buffer -- a demonstration, not a licence: a real out-of-bounds vertex fetch is undefined
// behaviour and can read another resource entirely.
static void make_geometry(Context& c) {
    std::vector<Vertex> v;
    std::vector<uint32_t> idx;
    build_cube(v, idx);
    c.cube_indices = uint32_t(idx.size());

    const VkBufferUsageFlags VB = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags IB = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const double t0 = now_us();
    make_buffer(c, c.cube_vb, v.size() * sizeof(Vertex), VB, false);
    upload_staged(c, c.cube_vb, v.data(), v.size() * sizeof(Vertex));
    make_buffer(c, c.cube_ib, idx.size() * 4, IB, false);
    upload_staged(c, c.cube_ib, idx.data(), idx.size() * 4);
    c.staged_us = now_us() - t0;

    std::printf("mesh     : %zu vertices (%zu bytes), %u indices, %d faces\n",
                v.size(), v.size() * sizeof(Vertex), c.cube_indices, 6);
    std::printf("upload   : both buffers staged in %.1f us\n", c.staged_us);
}

// WHAT EVERY SCENE DRAWS, WHICH TODAY IS THE SAME THING.
struct Draw { VkBuffer vb, ib; uint32_t count, first; };

static Draw draw_for(const Context& c, const Scene&) {
    Draw d = { c.cube_vb.handle, c.cube_ib.handle, c.cube_indices, 0 };
    return d;
}

// ---- THE SIXTEEN FLOATS A SCENE COMES DOWN TO -----------------------------------------------
// model, then view, then projection -- and the multiplication reads RIGHT TO LEFT: the vertex
// meets the model matrix first. Writing it the other way round is the single most common
// matrix bug there is, and it produces a picture rather than an error.
static Mat4 mvp_for(const Scene& s, float ox, float oy, float oz) {
    const float PI = 3.14159265358979f;
    const Mat4 model = mul(translate(ox, oy, oz), mul(rotate_y(s.yaw), rotate_x(s.pitch)));
    const Mat4 view = look_at(0.0f, 0.0f, s.distance);
    const float aspect = float(RENDER_W) / float(RENDER_H);
    const Mat4 proj = s.proj == PROJ_ORTHO
        ? ortho(s.fov_deg, aspect, 0.1f, 100.0f, s.flip)
        : perspective(s.fov_deg * PI / 180.0f, aspect, 0.1f, 100.0f, s.flip);
    return mul(proj, mul(view, model));
}

// =======================================================================================
// THE PIPELINE.  Nine structs, and it cannot be changed afterwards.
// =======================================================================================
// Everything that was a call in OpenGL and a state object in Direct3D 11 is a FIELD here, and
// all of it is frozen when the pipeline is created: the shaders, the topology, the cull mode,
// the blend state, the viewport, the scissor, the format it will render into.
//
// That is not bureaucracy. It is what lets the driver compile the shaders knowing exactly what
// they will be used with, once, at load time, instead of guessing at the first draw. The price
// is that a different viewport is, strictly, a different pipeline -- which is why Day 5 is
// about dynamic state.
static VkPipeline make_pipeline(const Context& c, const Scene& s, VkFormat format) {
    // ---- THE THREE CONSTANTS THIS PIPELINE IS COMPILED WITH --------------------------------
    // A VkSpecializationInfo is a blob of bytes and a map saying which constant_id starts
    // where. The driver substitutes them into the SPIR-V and compiles it again, so a scene
    // with a different segment count is a different pipeline -- and the shader's loop bound is
    // a literal rather than a uniform read.
    // THE VERTEX STAGE HAS NO CONSTANTS ANY MORE. Everything it used to be compiled with --
    // the aspect ratio, the point size -- is now inside the matrix or gone, which is one fewer
    // reason for two scenes to need two pipelines.

    // THE FRAGMENT STAGE STILL HAS ONE. A blended scene's alpha is compiled into the
    // shader rather than passed at draw time, which is why changing it is a new pipeline.
    const VkSpecializationMapEntry fentry = { 0, 0, sizeof(float) };
    VkSpecializationInfo fsi = {};
    fsi.mapEntryCount = 1;
    fsi.pMapEntries = &fentry;
    fsi.dataSize = sizeof(float);
    static const float kOpaque = 1.0f;   // not OPAQUE: wingdi.h already has that
    fsi.pData = &kOpaque;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_mesh;
    stages[0].pName = "main";           // the SPIR-V entry point, and it need not be "main"
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = c.fs;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = &fsi;

    // ---- THE VERTEX INPUT STATE, UNCHANGED IN ALL TEN SCENES -----------------------------
    // Yesterday this was the variable and the pipeline was the constant. Today it is the other
    // way round: one interleaved buffer, one binding, two attributes, and every difference
    // between the ten pictures is a field further down this function.
    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    // LOCATION 0 IS THREE FLOATS NOW. One enum, and it is the whole of "the engine is 3D".
    const VkVertexInputAttributeDescription attrs[2] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x) },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r) },
    };

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = attrs;

    // THE TOPOLOGY IS FIXED TODAY AND THE POLYGON MODE IS NOT. They are easy to confuse: the
    // topology says how vertices group into primitives, the polygon mode says how a TRIANGLE
    // is filled in. Wireframe is a triangle list drawn as lines, not a line list.
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // ---- THE STATE THAT IS NOT BAKED IN -----------------------------------------------
    // VIEWPORT AND SCISSOR ARE DYNAMIC IN EVERY PIPELINE THIS ENGINE MAKES. It is what every
    // real engine does, and the reason is not speed: it is that a resized window otherwise
    // invalidates every pipeline in the program. The COUNTS are still baked in; only the
    // values move to the command buffer.
    //
    // LINE WIDTH IS DYNAMIC TOO, which makes "wireframe" and "widelines" -- identical in every
    // other field -- the only two scenes today that could share one pipeline. They do not,
    // because the engine builds one per scene, and Day 16 is where that starts to matter.
    const VkDynamicState dynamics[3] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 3;
    dyn.pDynamicStates = dynamics;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = nullptr;      // supplied by vkCmdSetViewport, not by this struct
    vps.scissorCount = 1;
    vps.pScissors = nullptr;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // THREE OPENGL CALLS, AS THREE FIELDS. glPolygonMode, glCullFace and glLineWidth were
    // things you did between draws; here they are things a pipeline IS.
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // CULLING IS ON FROM TODAY, and it is not a demonstration any more: the cube is a closed
    // solid, so every triangle facing away is one the camera cannot reach. Half the geometry
    // in the scene stops reaching the fragment shader, for one enum.
    rs.cullMode = s.cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;                // ignored: LINE_WIDTH is in the dynamic list

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ---- BLENDING IS PER ATTACHMENT, AND IT IS SIX ENUMS ---------------------------------
    // glEnable(GL_BLEND) plus glBlendFunc, as a struct: a factor for the source, a factor for
    // the destination, and an operation, separately for colour and for alpha.
    VkPipelineColorBlendAttachmentState cba = {};
    cba.blendEnable = VK_FALSE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // WITH DYNAMIC RENDERING THE PIPELINE IS TOLD A FORMAT, not a render pass. In Vulkan 1.0
    // this was a VkRenderPass object that had to exist first and had to be compatible; Day 16
    // explains what that bought and where it still matters.
    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &format;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = c.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    // THE SECOND PARAMETER IS THE CACHE, and passing VK_NULL_HANDLE there is what makes a cold
    // creation cold. It is not a hint: a cache that already holds this pipeline's compiled form
    // turns most of this call into a lookup.
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipeline));
    return pipeline;
}

// =======================================================================================
// A WINDOW, AND THE FOUR OBJECTS BETWEEN IT AND A PICTURE.
// A COMPARISON THE ENGINE MAKES ABOUT ITS OWN OUTPUT. The chapter claims that halving the
// render size changes a certain fraction of the frame. A claim like that is worth nothing
// unless the program that made both pictures is the thing that counted, so this counts -- on
// the RGB channels, exactly, with no tolerance.
static double fraction_differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return 1.0;
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i += 4)
        if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2]) ++n;
    return double(n) / double(a.size() / 4);
}

// =======================================================================================
// =======================================================================================
static LRESULT CALLBACK window_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE || (m == WM_KEYDOWN && w == VK_ESCAPE)) { PostQuitMessage(0); return 0; }
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}

struct Window {
    HWND             hwnd = nullptr;
    VkSurfaceKHR     surface = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain = VK_NULL_HANDLE;
    VkFormat         format = VK_FORMAT_UNDEFINED;
    VkExtent2D       extent = {};
    std::vector<VkImage> images;
    VkSemaphore      acquired = VK_NULL_HANDLE;   // the image is free to write
    VkSemaphore      finished = VK_NULL_HANDLE;   // the frame is done and may be shown
};

// THE NAME OF EVERY PRESENT MODE, because the capture prints what this machine offers and a
// number is not an answer.
static const char* present_mode_name(VkPresentModeKHR m) {
    switch (m) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE   tear, never wait";
    case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX     replace the queued frame";
    case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO        a queue, and the only one";
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED  tear only if late";
    default:                               return "other";
    }
}

static bool make_window(Context& c, Window& w, uint32_t width, uint32_t height, bool show) {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = window_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"MeridianWindow";
    RegisterClassExW(&wc);

    RECT r = { 0, 0, (LONG)width, (LONG)height };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    w.hwnd = CreateWindowExW(0, wc.lpszClassName, L"Meridian", WS_OVERLAPPEDWINDOW,
                             CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right - r.left, r.bottom - r.top,
                             nullptr, nullptr, wc.hInstance, nullptr);
    if (!w.hwnd) return false;
    if (show) ShowWindow(w.hwnd, SW_SHOW);

    // ---- the surface, and the only Windows-specific Vulkan in this book --------------------
    VkWin32SurfaceCreateInfoKHR sci = {};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = wc.hInstance;
    sci.hwnd = w.hwnd;
    VK_CHECK(vkCreateWin32SurfaceKHR(c.instance, &sci, nullptr, &w.surface));

    // A QUEUE FAMILY THAT CAN DRAW IS NOT NECESSARILY ONE THAT CAN PRESENT, and the only way
    // to know is to ask about this surface. On this card family 0 can do both; on some
    // hardware the answer is two different families and every frame crosses between them.
    VkBool32 can_present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(c.gpu, c.family, w.surface, &can_present);
    std::printf("surface  : queue family %u %s present to this window\n",
                c.family, can_present ? "CAN" : "CANNOT");
    if (!can_present) return false;

    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c.gpu, w.surface, &caps);
    std::printf("surface  : %u to %u image(s), current %ux%u\n",
                caps.minImageCount, caps.maxImageCount,
                caps.currentExtent.width, caps.currentExtent.height);

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.gpu, w.surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.gpu, w.surface, &nf, formats.data());
    std::printf("surface  : %u format(s) offered\n", nf);
    w.format = formats[0].format;
    VkColorSpaceKHR space = formats[0].colorSpace;
    for (const VkSurfaceFormatKHR& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { w.format = f.format; space = f.colorSpace; }

    uint32_t nm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.gpu, w.surface, &nm, nullptr);
    std::vector<VkPresentModeKHR> modes(nm);
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.gpu, w.surface, &nm, modes.data());
    std::printf("present  : %u mode(s) on this surface\n", nm);
    for (VkPresentModeKHR m : modes) std::printf("  %s\n", present_mode_name(m));

    // FIFO IS THE ONLY MODE EVERY IMPLEMENTATION MUST SUPPORT. It is a queue: one frame a
    // refresh, no tearing, and the presentation engine blocks you when it is full. Everything
    // else is a negotiation, and FIFO is the one you can rely on.
    w.extent = caps.currentExtent;
    VkSwapchainCreateInfoKHR ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = w.surface;
    ci.minImageCount = caps.minImageCount + 1 <= caps.maxImageCount || caps.maxImageCount == 0
                     ? caps.minImageCount + 1 : caps.minImageCount;
    ci.imageFormat = w.format;
    ci.imageColorSpace = space;
    ci.imageExtent = w.extent;
    ci.imageArrayLayers = 1;
    // TRANSFER_DST, NOT COLOR_ATTACHMENT. Nothing is ever drawn into a swapchain image here;
    // a finished picture is copied in. That is Day 1's decision, still holding.
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(c.device, &ci, nullptr, &w.swapchain));

    uint32_t ni = 0;
    vkGetSwapchainImagesKHR(c.device, w.swapchain, &ni, nullptr);
    w.images.resize(ni);
    vkGetSwapchainImagesKHR(c.device, w.swapchain, &ni, w.images.data());
    std::printf("swapchain: %u image(s) of %ux%u, FIFO, and you do not own them\n",
                ni, w.extent.width, w.extent.height);

    VkSemaphoreCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(c.device, &si, nullptr, &w.acquired));
    VK_CHECK(vkCreateSemaphore(c.device, &si, nullptr, &w.finished));
    return true;
}

static void destroy_window(Context& c, Window& w) {
    if (w.acquired) vkDestroySemaphore(c.device, w.acquired, nullptr);
    if (w.finished) vkDestroySemaphore(c.device, w.finished, nullptr);
    if (w.swapchain) vkDestroySwapchainKHR(c.device, w.swapchain, nullptr);
    if (w.surface) vkDestroySurfaceKHR(c.instance, w.surface, nullptr);
    if (w.hwnd) DestroyWindow(w.hwnd);
}

// ONE SCENE: DRAW INTO A SOURCE, COPY INTO A DESTINATION, READ THE DESTINATION.
// =======================================================================================
// THE SECOND IMAGE IS THE POINT. Until today the image drawn into was the image written out.
// Now there are two, and the copy between them is vkCmdBlitImage -- which is what the window
// gets as well, with a swapchain image in place of the destination.
static bool render_scene(Context& c, const Scene& s, std::vector<uint8_t>& out, bool* swapped) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;

    auto make_image = [&](uint32_t w, uint32_t h, VkImageUsageFlags usage,
                          VkImage& image, VkDeviceMemory& mem, VkImageView* view) {
        VkImageCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = FORMAT;
        ici.extent = { w, h, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &image));
        VkMemoryRequirements mr = {};
        vkGetImageMemoryRequirements(c.device, image, &mr);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &mem));
        VK_CHECK(vkBindImageMemory(c.device, image, mem, 0));
        if (view) {
            VkImageViewCreateInfo ivi = {};
            ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivi.image = image;
            ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivi.format = FORMAT;
            ivi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(c.device, &ivi, nullptr, view));
        }
    };

    VkImage src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    VkDeviceMemory src_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;
    VkImageView src_view = VK_NULL_HANDLE;
    make_image(RENDER_W, RENDER_H,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               src, src_mem, &src_view);
    // THE DESTINATION IS A TRANSFER TARGET AND NOTHING IS DRAWN INTO IT. That is exactly what
    // a swapchain image is: something you copy a finished picture into.
    make_image(RENDER_W, RENDER_H,
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               dst, dst_mem, nullptr);

    const VkDeviceSize bytes = VkDeviceSize(RENDER_W) * RENDER_H * 4;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer readback = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(c.device, &bci, nullptr, &readback));
    VkMemoryRequirements br = {};
    vkGetBufferMemoryRequirements(c.device, readback, &br);
    VkMemoryAllocateInfo bai = {};
    bai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memory_type(c.gpu, br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(c.device, &bai, nullptr, &readback_mem));
    VK_CHECK(vkBindBufferMemory(c.device, readback, readback_mem, 0));

    VkPipeline pipeline = make_pipeline(c, s, FORMAT);

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    auto barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 ss, VkAccessFlags2 sa,
                       VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = ss; b.srcAccessMask = sa;
        b.dstStageMask = ds; b.dstAccessMask = da;
        b.oldLayout = from; b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    barrier(src, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo att = {};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = src_view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    std::memcpy(att.clearValue.color.float32, NIGHT, sizeof NIGHT);

    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { { 0, 0 }, { RENDER_W, RENDER_H } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // ---- BINDING THE BYTES, AND THEN ASKING FOR THEM ---------------------------------------
    // vkCmdBindVertexBuffers takes an ARRAY of buffers and an array of offsets, because a
    // pipeline can have several bindings. The offset is why one big buffer can hold every mesh
    // in a scene and each draw simply starts further in -- which is what Day 10 does.
    const Draw d = draw_for(c, s);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &d.vb, &off);
    vkCmdBindIndexBuffer(cmd, d.ib, 0, VK_INDEX_TYPE_UINT32);

    // ---- ONE VIEWPORT, AND THEN A MATRIX PER OBJECT ---------------------------------------
    VkViewport vp = { 0.0f, 0.0f, float(RENDER_W), float(RENDER_H), 0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sr = { { 0, 0 }, { RENDER_W, RENDER_H } };
    vkCmdSetScissor(cmd, 0, 1, &sr);
    vkCmdSetLineWidth(cmd, 1.0f);

    // A LATTICE, OR ONE. Twenty-seven cubes are twenty-seven draws, and the ONLY thing that
    // differs between them is sixty-four bytes recorded into the command buffer -- no buffer
    // update, no descriptor, no barrier, nothing to synchronise.
    const int g = s.grid;
    const float spread = 2.6f;
    for (int gz = 0; gz < g; ++gz)
        for (int gy = 0; gy < g; ++gy)
            for (int gx = 0; gx < g; ++gx) {
                const float o = (g - 1) * 0.5f;
                const Mat4 mvp = mvp_for(s, (gx - o) * spread, (gy - o) * spread,
                                         (gz - o) * spread);
                vkCmdPushConstants(cmd, c.layout, VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof mvp, &mvp);
                vkCmdDrawIndexed(cmd, d.count, 1, d.first, 0, 0);
            }
    vkCmdEndRendering(cmd);

    barrier(src, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier(dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // THE BARS IN A LETTERBOX ARE NOT DRAWN. They are whatever the destination already held,
    // so it is cleared first and the copy lands in the middle of it. A swapchain image is the
    // same: what you do not write is what the last frame left.
    VkClearColorValue bar = {};
    bar.float32[0] = 0.02f; bar.float32[1] = 0.02f; bar.float32[2] = 0.03f;
    bar.float32[3] = 1.0f;
    VkImageSubresourceRange whole = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bar, 1, &whole);

    const Rect r = fit_into(FIT_FILL, RENDER_W, RENDER_H, RENDER_W, RENDER_H);
    // A BLIT NAMES TWO RECTANGLES AND NOTHING REQUIRES THEM TO MATCH. Swapping the
    // destination's y values is how a copy is flipped, and forgetting that you have already
    // flipped once is how a picture ends up upside down.
    VkImageBlit2 region = {};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[0] = { 0, 0, 0 };
    region.srcOffsets[1] = { int32_t(RENDER_W), int32_t(RENDER_H), 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[0] = { r.x0, r.y0, 0 };
    region.dstOffsets[1] = { r.x1, r.y1, 1 };

    VkBlitImageInfo2 blit = {};
    blit.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blit.srcImage = src;
    blit.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit.dstImage = dst;
    blit.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blit.regionCount = 1;
    blit.pRegions = &region;
    blit.filter = VK_FILTER_LINEAR;
    vkCmdBlitImage2(cmd, &blit);

    barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region2 = {};
    region2.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region2.imageExtent = { RENDER_W, RENDER_H, 1 };
    vkCmdCopyImageToBuffer(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region2);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(c.device, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    out.assign((const uint8_t*)mapped, (const uint8_t*)mapped + bytes);
    wchar_t path[256];
    swprintf(path, 256, L"frame_%S.png", s.name);
    const bool ok = write_png(path, out.data(), RENDER_W, RENDER_H, RENDER_W * 4, swapped);
    vkUnmapMemory(c.device, readback_mem);

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    vkDestroyPipeline(c.device, pipeline, nullptr);
    vkDestroyBuffer(c.device, readback, nullptr);
    vkFreeMemory(c.device, readback_mem, nullptr);
    vkDestroyImageView(c.device, src_view, nullptr);
    vkDestroyImage(c.device, src, nullptr);
    vkFreeMemory(c.device, src_mem, nullptr);
    vkDestroyImage(c.device, dst, nullptr);
    vkFreeMemory(c.device, dst_mem, nullptr);
    return ok;
}

// =======================================================================================
// WHAT A PIPELINE COSTS, AND WHAT THE CACHE TAKES IT DOWN TO.
// =======================================================================================
// Three passes over the same ten scenes. The first has no cache at all, which is the price a
// program pays on a machine it has never run on. The second fills a cache. The third uses it.
//
// THE CACHE IS NOT A SPEED SETTING. It is a blob of driver-specific bytes you are expected to
// write to disk and hand back next launch -- which is what "shader compilation" progress bars
// in shipped games are doing, and why they only appear once.
static void measure_pipelines(Context& c) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;

    c.cache = VK_NULL_HANDLE;
    double t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double cold = now_us() - t;

    VkPipelineCacheCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK(vkCreatePipelineCache(c.device, &pci, nullptr, &c.cache));

    t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double filling = now_us() - t;

    t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double hot = now_us() - t;

    size_t bytes = 0;
    vkGetPipelineCacheData(c.device, c.cache, &bytes, nullptr);

    std::printf("pipeline : %d scene(s), each one a separate compiled object\n", SCENE_COUNT);
    std::printf("pipeline : cold %7.0f us   filling %7.0f us   cached %7.0f us\n",
                cold, filling, hot);
    std::printf("pipeline : %.0f us each cold, %.0f us each cached\n",
                cold / SCENE_COUNT, hot / SCENE_COUNT);
    std::printf("cache    : %zu bytes, and this is what a game writes to disk\n", bytes);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    const char* want = nullptr;
    bool provoke = false, window_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) want = argv[++i];
        else if (std::strcmp(argv[i], "--provoke") == 0) provoke = true;
        else if (std::strcmp(argv[i], "--window") == 0) window_mode = true;
    }
    // A full run always ends with the provocation, so one capture holds the whole day --
    // including the validation layer's account of a real mistake.
    if (!want) provoke = true;

    Context c;
    make_context(c, true);

    // ---- THE MEMORY TABLE THIS CARD ACTUALLY OFFERS ---------------------------------------
    // Not an UPLOAD heap and a DEFAULT heap. A list, with flags, and the choice is yours.
    VkPhysicalDeviceMemoryProperties mp = {};
    vkGetPhysicalDeviceMemoryProperties(c.gpu, &mp);
    std::printf("memory   : %u type(s) in %u heap(s)\n", mp.memoryTypeCount, mp.memoryHeapCount);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        std::printf("  heap %u  %6llu MB  %s\n", i,
                    (unsigned long long)(mp.memoryHeaps[i].size / (1024 * 1024)),
                    (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                        ? "device local" : "host");
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        std::printf("  type %2u heap %u  %s%s%s%s\n", i, mp.memoryTypes[i].heapIndex,
                    (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED" : "");
    }

    // ---- THE BUFFERS, BUILT ONCE ----------------------------------------------------------
    make_geometry(c);
    measure_pipelines(c);
    std::printf("memtype  : cube_vb in type %u, cube_ib in type %u\n",
                c.cube_vb.type_index, c.cube_ib.type_index);

    // ---- THE FLIP, AS A NUMBER RATHER THAN AN ASSERTION -----------------------------------
    // The chapter claims one element of the projection matrix changes sign. This prints it,
    // from the same function the renderer calls, so the claim cannot drift away from the code.
    const Mat4 up = mvp_for(SCENES[0], 0, 0, 0);
    const Mat4 down = mvp_for(SCENES[5], 0, 0, 0);
    std::printf("matrix   : mvp[5] with the flip %+.4f, without it %+.4f\n", up.m[5], down.m[5]);
    std::printf("matrix   : every other element identical: %s\n",
                (up.m[0] == down.m[0] && up.m[10] == down.m[10] && up.m[11] == down.m[11])
                    ? "yes" : "no");

    // ---- A WINDOW, IF ASKED FOR, AND THE WHOLE OF WHAT IT ADDS ---------------------------
    // It is created even for a capture, and NOT SHOWN. Creating a surface is what proves the
    // extensions work and what makes the capture report what this machine offers -- and a
    // window nobody sees cannot land in a screenshot, which is this book's rule.
    Window win;
    if (!make_window(c, win, 1280, 720, window_mode)) {
        std::printf("FAILED: no window or no surface\n");
        return 1;
    }
    if (!window_mode)
        std::printf("window   : created, never shown -- the capture opens no screen\n");

    if (window_mode) {
        // ONE FRAME, PRESENTED, AND THE FOUR STEPS EVERY VULKAN FRAME HAS.
        std::printf("window   : presenting until Esc\n");
        std::vector<uint8_t> px;
        bool sw = false;
        const Scene& s = SCENES[0];
        (void)render_scene(c, s, px, &sw);
        MSG msg = {};
        bool running = true;
        while (running) {
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) running = false;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!running) break;
            uint32_t index = 0;
            const VkResult acq = vkAcquireNextImageKHR(c.device, win.swapchain, UINT64_MAX,
                                                       win.acquired, VK_NULL_HANDLE, &index);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) break;
            VkPresentInfoKHR pi = {};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &win.acquired;
            pi.swapchainCount = 1;
            pi.pSwapchains = &win.swapchain;
            pi.pImageIndices = &index;
            if (vkQueuePresentKHR(c.queue, &pi) != VK_SUCCESS) break;
            vkQueueWaitIdle(c.queue);
        }
        vkDeviceWaitIdle(c.device);
        destroy_window(c, win);
        vkDestroyPipelineLayout(c.device, c.layout, nullptr);
        vkDestroyShaderModule(c.device, c.vs_mesh, nullptr);
        vkDestroyShaderModule(c.device, c.fs, nullptr);
        vkDestroyCommandPool(c.device, c.pool, nullptr);
        vkDestroyDevice(c.device, nullptr);
        vkDestroyInstance(c.instance, nullptr);
        CoUninitialize();
        return 0;
    }

    bool swapped = false;
    int made = 0;
    std::vector<uint8_t> px;
    std::map<std::string, std::vector<uint8_t>> kept;
    for (const Scene& s : SCENES) {
        if (want && std::strcmp(want, s.name) != 0) continue;
        if (!render_scene(c, s, px, &swapped)) {
            std::printf("FAILED: writing frame_%s.png\n", s.name);
            return 1;
        }
        // The frames are kept so the comparison below is made from the same bytes that
        // went into the PNG rather than from a second render.
        kept[s.name] = px;
        const Draw d = draw_for(c, s);
        // THE LINE FITS EIGHTY-EIGHT COLUMNS, which is the width of a printed listing. A
        // capture line wider than the measure is a line this book cannot quote.
        std::printf("wrote  : %-9s %3d draw(s)  %s\n", s.name, s.grid * s.grid * s.grid,
                    s.note);
        (void)d;
        ++made;
    }

    // ---- WHAT READING THE SAME BYTES DIFFERENTLY COST -------------------------------------
    // Every scene is 1920x1080, so all of them compare pixel for pixel. The numbers the chapter
    // prints are these numbers, and if the engine changes they change.
    if (!want && !kept["cube"].empty()) {
        struct Pair { const char* a; const char* b; };
        static const Pair PAIRS[] = {
            { "spin",   "cube"   },
            { "wide",   "cube"   },
            { "narrow", "cube"   },
            { "ortho",  "cube"   },
            { "noflip", "cube"   },
            { "nocull", "cube"   },
            { "close",  "wide"   },
            { "grid",   "cube"   },
        };
        for (const Pair& pr : PAIRS) {
            if (kept[pr.a].empty() || kept[pr.b].empty()) continue;
            std::printf("compare: %-9s vs %-7s %6.2f%% of pixels differ\n",
                        pr.a, pr.b, 100.0 * fraction_differing(kept[pr.a], kept[pr.b]));
        }
    }

    if (provoke) {
        // A DELIBERATE MISTAKE, so the layer can be seen doing its job. A command buffer that
        // is submitted twice without being reset breaks a rule the driver will not mention.
        std::printf("provoke  : submitting a command buffer that was never recorded\n");
        VkCommandBufferAllocateInfo cbi = {};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = c.pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        VkCommandBuffer empty = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(c.device, &cbi, &empty);

        VkCommandBufferSubmitInfo csi = {};
        csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        csi.commandBuffer = empty;
        VkSubmitInfo2 si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &csi;
        const VkResult r = vkQueueSubmit2(c.queue, 1, &si, VK_NULL_HANDLE);
        std::printf("provoke  : vkQueueSubmit2 returned %d, and the layer said the rest\n",
                    (int)r);
        vkQueueWaitIdle(c.queue);
        vkFreeCommandBuffers(c.device, c.pool, 1, &empty);
    }

    std::printf("wic      : asked for RGBA and was given %s\n", swapped ? "BGRA" : "RGBA");
    std::printf("debug    : %d message(s) from the validation layer this run\n", g_messages);
    std::printf("done     : %d picture(s), no window opened\n", made);

    destroy_buffer(c, c.cube_vb);
    destroy_buffer(c, c.cube_ib);
    if (c.cache) vkDestroyPipelineCache(c.device, c.cache, nullptr);
    destroy_window(c, win);
    vkDestroyPipelineLayout(c.device, c.layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_mesh, nullptr);
    vkDestroyShaderModule(c.device, c.fs, nullptr);
    vkDestroyCommandPool(c.device, c.pool, nullptr);
    vkDestroyDevice(c.device, nullptr);
    if (c.messenger) {
        auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(c.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(c.instance, c.messenger, nullptr);
    }
    vkDestroyInstance(c.instance, nullptr);
    CoUninitialize();
    return 0;
}
