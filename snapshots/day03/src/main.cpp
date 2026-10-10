// =======================================================================================
// MERIDIAN - Day 3.  A window, a surface, and the swapchain.
//
// THE PICTURE HAS NOT CHANGED AND THE ENGINE HAS. For two days Meridian has rendered into an
// image it owns and written a PNG. Today a window appears beside that, and the picture is
// COPIED into it -- which is the arrangement the rest of the book keeps, and the reason Day 1
// was allowed to end in a file.
//
// FOUR OBJECTS, AND THEY ARE ALL EXTENSIONS.
//
//   VkSurfaceKHR      what Vulkan calls a window it might present to. Created from an HWND by
//                     VK_KHR_win32_surface, and the only Windows-specific Vulkan in this book.
//   VkSwapchainKHR    a small ring of images owned by the presentation engine, not by you.
//   VkSemaphore       how the queue waits for an image to be free, and the window for a frame
//                     to be finished. Not a fence: a fence talks to the CPU, this does not.
//   the present mode  FIFO, MAILBOX or IMMEDIATE -- three answers to what happens when you
//                     draw faster than the screen refreshes.
//
// NONE OF IT IS CORE VULKAN. Presentation is an extension because Vulkan runs on machines with
// no screen at all, and a compute job on a server should not have to pretend otherwise. Asking
// for VK_KHR_swapchain is asking for something the specification treats as optional -- which
// is exactly what it is.
//
// AND THE COPY IS vkCmdBlitImage, which names TWO rectangles and does not require them to
// match. Different sizes resize, different shapes stretch, and reversed corners flip. Today's
// ten scenes are that one call, asked ten different questions, with the engine measuring what
// each of them cost in pixels.
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

// ---- THE SCENES -----------------------------------------------------------------------
// Ten pictures, each one thing different, and every figure in this chapter is one of them.
// A table rather than ten functions, because the differences ARE the chapter.
// HOW A SOURCE RECTANGLE IS FITTED INTO A DESTINATION. The three answers a real engine
// needs, and Day 3 photographs all of them.
enum Fit { FIT_FILL, FIT_LETTERBOX, FIT_STRETCH };

struct Scene {
    const char*   name;
    uint32_t      src_w, src_h;   // where the fan is drawn
    uint32_t      dst_w, dst_h;   // where it is copied to, and what the PNG is
    Fit           fit;
    VkFilter      filter;
    bool          flip;           // reverse the destination's y, as a swapchain never would
    const char*   note;
};


static const Scene SCENES[] = {
    { "direct",      1920, 1080, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  false,
      "no resize: one pixel to one pixel"          },
    { "half",         960,  540, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  false,
      "half size, magnified smoothly"              },
    { "nearest",      960,  540, 1920, 1080, FIT_FILL,      VK_FILTER_NEAREST, false,
      "half size, magnified blockily"              },
    { "quarter",      480,  270, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  false,
      "a quarter of the width, four times up"      },
    { "eighth",       240,  135, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  false,
      "an eighth: the pixels ARE the picture"      },
    { "supersample", 3840, 2160, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  false,
      "twice the size, shrunk back down"           },
    { "letterbox",   1920, 1080, 1440, 1080, FIT_LETTERBOX, VK_FILTER_LINEAR,  false,
      "a 4:3 window, shape kept"                   },
    { "stretched",   1920, 1080, 1440, 1080, FIT_STRETCH,   VK_FILTER_LINEAR,  false,
      "a 4:3 window, shape given up"               },
    { "flipped",     1920, 1080, 1920, 1080, FIT_FILL,      VK_FILTER_LINEAR,  true,
      "the destination's y values swapped"         },
    { "tall",        1920, 1080,  810, 1080, FIT_LETTERBOX, VK_FILTER_LINEAR,  false,
      "a narrow window: bars at the sides"         },
};
static const int SCENE_COUNT = int(sizeof SCENES / sizeof SCENES[0]);

// The fan is the same every scene now; only the copy changes.
static const int FAN_SEGMENTS = 96;

struct Rect { int32_t x0, y0, x1, y1; };

// WHERE A SOURCE OF ONE SHAPE LANDS INSIDE A DESTINATION OF ANOTHER. Three answers, and every
// engine with a window has to pick one: fill it and accept the distortion, keep the shape and
// leave bars, or crop. Meridian letterboxes, which is what a game does.
static Rect fit_into(Fit how, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    if (how == FIT_STRETCH || how == FIT_FILL)
        return { 0, 0, int32_t(dw), int32_t(dh) };
    const double sa = double(sw) / double(sh), da = double(dw) / double(dh);
    if (da > sa) {                       // the destination is wider: bars at the sides
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
    VkShaderModule   vs_fan = VK_NULL_HANDLE;
    VkShaderModule   fs = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
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

    c.vs_fan = make_module(c.device, kFanVert, sizeof kFanVert);
    c.fs = make_module(c.device, kTriFrag, sizeof kTriFrag);
    std::printf("spirv    : %zu and %zu words, compiled at BUILD time\n",
                sizeof kFanVert / 4, sizeof kTriFrag / 4);

    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.layout));
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
    struct Spec { int32_t segments; int32_t mode; float aspect; } spec = {
        FAN_SEGMENTS, 0, float(s.src_w) / float(s.src_h) };
    const VkSpecializationMapEntry entries[3] = {
        { 0, offsetof(Spec, segments), sizeof(int32_t) },
        { 1, offsetof(Spec, mode),     sizeof(int32_t) },
        { 2, offsetof(Spec, aspect),   sizeof(float)   },
    };
    VkSpecializationInfo si = {};
    si.mapEntryCount = 3;
    si.pMapEntries = entries;
    si.dataSize = sizeof spec;
    si.pData = &spec;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_fan;
    stages[0].pName = "main";           // the SPIR-V entry point, and it need not be "main"
    stages[0].pSpecializationInfo = &si;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = c.fs;
    stages[1].pName = "main";

    // NO VERTEX INPUT AT ALL. The shader builds its three corners from gl_VertexIndex, so
    // there is no buffer, no binding and no attribute -- and this struct is still required.
    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp = { 0.0f, 0.0f, float(s.src_w), float(s.src_h), 0.0f, 1.0f };
    VkRect2D sr = { { 0, 0 }, { s.src_w, s.src_h } };

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sr;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;    // Day 6 turns this on, when there is a back to cull
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {};
    cba.blendEnable = VK_FALSE;
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
    gpi.layout = c.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(c.device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
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
    make_image(s.src_w, s.src_h,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               src, src_mem, &src_view);
    // THE DESTINATION IS A TRANSFER TARGET AND NOTHING IS DRAWN INTO IT. That is exactly what
    // a swapchain image is: something you copy a finished picture into.
    make_image(s.dst_w, s.dst_h,
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               dst, dst_mem, nullptr);

    const VkDeviceSize bytes = VkDeviceSize(s.dst_w) * s.dst_h * 4;
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
    ri.renderArea = { { 0, 0 }, { s.src_w, s.src_h } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for (int i = 0; i < FAN_SEGMENTS; ++i)
        vkCmdDraw(cmd, 3, 1, uint32_t(i * 3), 0);
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

    const Rect r = fit_into(s.fit, s.src_w, s.src_h, s.dst_w, s.dst_h);
    // A BLIT NAMES TWO RECTANGLES AND NOTHING REQUIRES THEM TO MATCH. Swapping the
    // destination's y values is how a copy is flipped, and forgetting that you have already
    // flipped once is how a picture ends up upside down.
    VkImageBlit2 region = {};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[0] = { 0, 0, 0 };
    region.srcOffsets[1] = { int32_t(s.src_w), int32_t(s.src_h), 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[0] = { r.x0, s.flip ? r.y1 : r.y0, 0 };
    region.dstOffsets[1] = { r.x1, s.flip ? r.y0 : r.y1, 1 };

    VkBlitImageInfo2 blit = {};
    blit.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blit.srcImage = src;
    blit.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit.dstImage = dst;
    blit.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blit.regionCount = 1;
    blit.pRegions = &region;
    blit.filter = s.filter;
    vkCmdBlitImage2(cmd, &blit);

    barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region2 = {};
    region2.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region2.imageExtent = { s.dst_w, s.dst_h, 1 };
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
    const bool ok = write_png(path, out.data(), s.dst_w, s.dst_h, s.dst_w * 4, swapped);
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

    VkPhysicalDeviceMemoryProperties mp = {};
    vkGetPhysicalDeviceMemoryProperties(c.gpu, &mp);
    std::printf("memory   : %u type(s) in %u heap(s)\n", mp.memoryTypeCount, mp.memoryHeapCount);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        std::printf("  heap %u  %6llu MB  %s\n", i,
                    (unsigned long long)(mp.memoryHeaps[i].size / (1024 * 1024)),
                    (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                        ? "device local" : "host");

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
        vkDestroyShaderModule(c.device, c.vs_fan, nullptr);
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
        // The full-size frames are kept so the comparison below is made from the same bytes
        // that went into the PNG rather than from a second render.
        if (s.dst_w == RENDER_W && s.dst_h == RENDER_H) kept[s.name] = px;
        // THE LINE FITS EIGHTY-EIGHT COLUMNS, which is the width of a printed listing. A
        // capture line wider than the measure is a line this book cannot quote.
        std::printf("wrote  : %-11s %4ux%-4u -> %4ux%-4u  %s\n",
                    s.name, s.src_w, s.src_h, s.dst_w, s.dst_h, s.note);
        ++made;
    }

    // ---- WHAT THE RESOLUTION DIAL ACTUALLY COST -------------------------------------------
    // Six of the scenes end at 1920x1080, so they can be compared pixel for pixel. The numbers
    // the chapter prints are these numbers, and if the engine changes they change.
    if (!want && !kept["direct"].empty()) {
        struct Pair { const char* a; const char* b; };
        static const Pair PAIRS[] = {
            { "half",        "direct" },
            { "nearest",     "half"   },
            { "quarter",     "direct" },
            { "supersample", "direct" },
            { "flipped",     "direct" },
        };
        for (const Pair& pr : PAIRS) {
            if (kept[pr.a].empty() || kept[pr.b].empty()) continue;
            std::printf("compare: %-11s vs %-7s %6.2f%% of pixels differ\n",
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

    destroy_window(c, win);
    vkDestroyPipelineLayout(c.device, c.layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_fan, nullptr);
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
