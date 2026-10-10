// =======================================================================================
// MERIDIAN - Day 2.  The loader, the ICD, and the trampoline you can skip.
//
// YESTERDAY'S PROGRAM CALLED vkCmdDraw AND SOMETHING ELSE ANSWERED. `vulkan-1.lib` does not
// contain Vulkan; it contains the LOADER, and the loader contains a trampoline for every
// command: a small function that reads a dispatch pointer out of the handle you passed and
// jumps to the driver's real one. Correct, necessary, and one indirection per call.
//
// THREE THINGS ARRIVE TODAY, and they are one idea.
//
//   WHAT THE LOADER ACTUALLY IS. vulkan-1.dll lives in System32 and was installed by your
//   graphics driver, not by Microsoft and not by the SDK. It finds the driver through a JSON
//   file named in the registry, and it finds LAYERS through the registry or through the
//   VK_LAYER_PATH environment variable -- which is why validation works on this machine
//   without anything having been installed at all.
//
//   A DISPATCH TABLE, GENERATED FROM vk.xml. The registry says which commands are
//   device-level, vkGetDeviceProcAddr returns the driver's own function for each, and
//   code/gen_dispatch.py writes the struct and a macro per command so that every call site in
//   this file reads exactly as it did yesterday.
//
//   AND THE DIFFERENCE, MEASURED. A hundred thousand recorded draws, through the loader and
//   through the table, timed. The number is at the end of the capture and it is not large --
//   which is worth knowing before you rewrite a renderer for it.
//
// THE PICTURE IS A FAN, and the fan is the point: every wedge is its own vkCmdDraw, so the
// 256-segment scene records 256 draw calls and the benchmark has something real to measure.
//
// AND SPECIALIZATION CONSTANTS, which is how one shader becomes ten pipelines. The segment
// count, the shape and the aspect ratio are constants inside the SPIR-V whose values are given
// when the pipeline is built -- so the loop bound is a literal at run time, and each scene's
// pipeline is a separate compilation of the same source.
//
//   meridian.exe               renders every scene below and exits
//   meridian.exe --all         the same thing, said out loud
//   meridian.exe --scene NAME  renders one of them
//   meridian.exe --provoke     break a rule on purpose, and print what validation says
// =======================================================================================
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#include <vulkan/vulkan.h>

#include "vk_dispatch.h"

#include <cstdio>
#include <cstring>
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
// THE SHAPES ONE SHADER CAN BE. Each is a specialization constant, so each is a separate
// compilation of fan.vert and a separate pipeline.
enum Shape { SHAPE_DISC = 0, SHAPE_STAR = 1, SHAPE_SPIRAL = 2 };

struct Scene {
    const char*   name;
    uint32_t      w, h;          // the image's size
    int           segments;      // wedges, and therefore draw calls
    Shape         shape;
    VkPolygonMode mode;
    float         line_width;    // pixels, and anything but 1.0 is a device feature
    const char*   note;
};

static const Scene SCENES[] = {
    { "fan",      1920, 1080,  64, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "64 wedges, and 64 draw calls"               },
    { "four",     1920, 1080,   4, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "four wedges: the construction, visible"     },
    { "eight",    1920, 1080,   8, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "eight, and the circle is still a polygon"   },
    { "sixteen",  1920, 1080,  16, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "sixteen, where the eye stops objecting"     },
    { "many",     1920, 1080, 256, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "256 wedges, and 256 trips through the loader" },
    { "star",     1920, 1080,  32, SHAPE_STAR,   VK_POLYGON_MODE_FILL,  1.0f,
      "one constant changed, and a different shape" },
    { "spiral",   1920, 1080, 128, SHAPE_SPIRAL, VK_POLYGON_MODE_FILL,  1.0f,
      "a radius that grows with the angle"         },
    { "wires",    1920, 1080,  24, SHAPE_DISC,   VK_POLYGON_MODE_LINE,  5.0f,
      "the same wedges, drawn as their edges"      },
    { "square",    900,  900,  64, SHAPE_DISC,   VK_POLYGON_MODE_FILL,  1.0f,
      "a square image, and the aspect constant 1.0" },
    { "wide",     2560,  720,  96, SHAPE_SPIRAL, VK_POLYGON_MODE_FILL,  1.0f,
      "a wide image, corrected by one constant"    },
};
static const int SCENE_COUNT = int(sizeof SCENES / sizeof SCENES[0]);

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
    const char* want_exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = validation ? &dbg : nullptr;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = validation ? 1u : 0u;
    ici.ppEnabledLayerNames = want_layers;
    ici.enabledExtensionCount = validation ? 1u : 0u;
    ici.ppEnabledExtensionNames = want_exts;
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

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
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
        s.segments, (int32_t)s.shape, float(s.w) / float(s.h) };
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

    VkViewport vp = { 0.0f, 0.0f, float(s.w), float(s.h), 0.0f, 1.0f };
    VkRect2D sr = { { 0, 0 }, { s.w, s.h } };

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sr;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = s.mode;
    rs.cullMode = VK_CULL_MODE_NONE;    // Day 6 turns this on, when there is a back to cull
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = s.line_width;        // anything but 1.0 needs the wideLines feature

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
// THE BENCHMARK'S OWN TARGET.  Sixty-four pixels, on a device with no layers on it.
// =======================================================================================
// It exists so that the draws being timed are LEGAL: inside a rendering, with a pipeline
// bound. A command that breaks a rule is not a command the driver handles the way it handles a
// real one, so timing an illegal draw measures nothing you can use.
struct BenchTarget {
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline     pipeline = VK_NULL_HANDLE;
};

static void make_bench_target(VkDevice bd, VkPhysicalDevice bpd, const Context&,
                              BenchTarget& t) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = FORMAT;
    ici.extent = { 64, 64, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK((::vkCreateImage)(bd, &ici, nullptr, &t.image));

    VkMemoryRequirements mr = {};
    (::vkGetImageMemoryRequirements)(bd, t.image, &mr);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(bpd, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK((::vkAllocateMemory)(bd, &mai, nullptr, &t.memory));
    VK_CHECK((::vkBindImageMemory)(bd, t.image, t.memory, 0));

    VkImageViewCreateInfo ivi = {};
    ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image = t.image;
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format = FORMAT;
    ivi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK((::vkCreateImageView)(bd, &ivi, nullptr, &t.view));

    auto module_on = [&](const uint32_t* words, size_t bytes) {
        VkShaderModuleCreateInfo smi = {};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = bytes;
        smi.pCode = words;
        VkShaderModule m = VK_NULL_HANDLE;
        VK_CHECK((::vkCreateShaderModule)(bd, &smi, nullptr, &m));
        return m;
    };
    t.vs = module_on(kFanVert, sizeof kFanVert);
    t.fs = module_on(kTriFrag, sizeof kTriFrag);

    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VK_CHECK((::vkCreatePipelineLayout)(bd, &pli, nullptr, &t.layout));

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = t.vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = t.fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp = { 0.0f, 0.0f, 64.0f, 64.0f, 0.0f, 1.0f };
    VkRect2D sr = { { 0, 0 }, { 64, 64 } };
    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sr;
    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba = {};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkFormat f = FORMAT;
    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &f;
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
    gpi.layout = t.layout;
    VK_CHECK((::vkCreateGraphicsPipelines)(bd, VK_NULL_HANDLE, 1, &gpi, nullptr, &t.pipeline));
}

static void destroy_bench_target(VkDevice bd, BenchTarget& t) {
    (::vkDestroyPipeline)(bd, t.pipeline, nullptr);
    (::vkDestroyPipelineLayout)(bd, t.layout, nullptr);
    (::vkDestroyShaderModule)(bd, t.vs, nullptr);
    (::vkDestroyShaderModule)(bd, t.fs, nullptr);
    (::vkDestroyImageView)(bd, t.view, nullptr);
    (::vkDestroyImage)(bd, t.image, nullptr);
    (::vkFreeMemory)(bd, t.memory, nullptr);
}

// =======================================================================================
// ONE SCENE, FROM AN EMPTY IMAGE TO A PNG.
// =======================================================================================
static bool render_scene(Context& c, const Scene& s, bool* swapped) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;

    // ---- the image, in memory on the card -------------------------------------------------
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = FORMAT;
    ici.extent = { s.w, s.h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    // OPTIMAL TILING MEANS "LAY IT OUT HOWEVER YOU LIKE". The card swizzles a texture into an
    // order that makes neighbouring texels neighbours in memory, and the CPU cannot read it.
    // That is why the finished picture is COPIED into a buffer rather than mapped in place.
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &image));
    VkMemoryRequirements mr = {};
    vkGetImageMemoryRequirements(c.device, image, &mr);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory image_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &image_mem));
    VK_CHECK(vkBindImageMemory(c.device, image, image_mem, 0));

    VkImageViewCreateInfo ivi = {};
    ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image = image;
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format = FORMAT;
    ivi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(c.device, &ivi, nullptr, &view));

    // ---- the buffer the picture comes home in ----------------------------------------------
    const VkDeviceSize bytes = VkDeviceSize(s.w) * s.h * 4;
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
    // HOST_VISIBLE so the CPU can map it; HOST_COHERENT so no flush is needed after the card
    // writes. Without coherent you must call vkInvalidateMappedMemoryRanges, and forgetting
    // that gives you a picture of whatever was in the buffer before.
    bai.memoryTypeIndex = memory_type(c.gpu, br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(c.device, &bai, nullptr, &readback_mem));
    VK_CHECK(vkBindBufferMemory(c.device, readback, readback_mem, 0));

    VkPipeline pipeline = make_pipeline(c, s, FORMAT);

    // ---- the command buffer -----------------------------------------------------------------
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

    // A BARRIER IS TWO STATEMENTS AT ONCE: what layout the image is moving between, and which
    // stages of which work have to finish before which other stages may start. Getting the
    // layout right and the stages wrong is the commonest Vulkan bug there is, and the only
    // thing that will tell you is the validation layer.
    auto barrier = [&](VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                       VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = src_stage;
        b.srcAccessMask = src_access;
        b.dstStageMask = dst_stage;
        b.dstAccessMask = dst_access;
        b.oldLayout = from;
        b.newLayout = to;
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

    // UNDEFINED is not a layout the image is in; it is a promise that you do not care what is
    // in it. Transitioning FROM it discards the contents, which is exactly right here because
    // the next thing that happens is a clear.
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo att = {};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    std::memcpy(att.clearValue.color.float32, NIGHT, sizeof NIGHT);

    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { { 0, 0 }, { s.w, s.h } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // ONE DRAW CALL PER WEDGE, ON PURPOSE. Three vertices each, with firstVertex stepping by
    // three so the shader can work out which wedge it is from gl_VertexIndex alone. A real
    // engine would draw all of them in one call; this one is counting trips through the
    // loader, and a trip you do not take cannot be measured.
    for (int i = 0; i < s.segments; ++i)
        vkCmdDraw(cmd, 3, 1, uint32_t(i * 3), 0);
    vkCmdEndRendering(cmd);

    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { s.w, s.h, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // ---- submit, and wait for the card ------------------------------------------------------
    // A FENCE IS THE CARD TELLING THE CPU IT HAS FINISHED. Nothing else does: vkQueueSubmit2
    // returns the moment the work is queued, not when it is done, and reading the buffer
    // before the fence is signalled reads whatever was there.
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

    // ---- the picture --------------------------------------------------------------------------
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(c.device, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    wchar_t path[256];
    swprintf(path, 256, L"frame_%S.png", s.name);
    const bool ok = write_png(path, (const uint8_t*)mapped, s.w, s.h, s.w * 4, swapped);
    vkUnmapMemory(c.device, readback_mem);

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    vkDestroyPipeline(c.device, pipeline, nullptr);
    vkDestroyBuffer(c.device, readback, nullptr);
    vkFreeMemory(c.device, readback_mem, nullptr);
    vkDestroyImageView(c.device, view, nullptr);
    vkDestroyImage(c.device, image, nullptr);
    vkFreeMemory(c.device, image_mem, nullptr);
    return ok;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    const char* want = nullptr;
    bool provoke = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) want = argv[++i];
        else if (std::strcmp(argv[i], "--provoke") == 0) provoke = true;
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

    bool swapped = false;
    int made = 0;
    for (const Scene& s : SCENES) {
        if (want && std::strcmp(want, s.name) != 0) continue;
        if (!render_scene(c, s, &swapped)) {
            std::printf("FAILED: writing frame_%s.png\n", s.name);
            return 1;
        }
        // THE LINE FITS EIGHTY-EIGHT COLUMNS, which is the width of a printed listing. A
        // capture line wider than the measure is a line this book cannot quote.
        std::printf("wrote  : %-9s %4ux%-4u %4d draws  %s\n",
                    s.name, s.w, s.h, s.segments, s.note);
        ++made;
    }

    // ---- WHAT THE TRAMPOLINE COSTS -----------------------------------------------------
    // AND IT HAS ITS OWN DEVICE, WITH NO LAYERS ON IT. The first version of this benchmark
    // ran on the engine's validation-enabled device and reported 2,703 nanoseconds a call.
    // That number is real and it is not the trampoline: it is the validation layer checking
    // every argument of every draw. A measurement of the loader has to be taken somewhere the
    // loader is the only thing in the way.
    //
    // THE DRAWS ARE ALSO LEGAL NOW. The first version recorded them outside any rendering and
    // with no pipeline bound, which the layer reported twenty times -- correctly, because a
    // driver is entitled to do anything at all with a command that breaks a rule, including
    // being faster than it would be with a real one.
    {
        VkApplicationInfo bapp = {};
        bapp.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        bapp.apiVersion = VK_API_VERSION_1_4;
        VkInstanceCreateInfo bici = {};
        bici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        bici.pApplicationInfo = &bapp;
        VkInstance bi = VK_NULL_HANDLE;
        VK_CHECK(vkCreateInstance(&bici, nullptr, &bi));

        uint32_t n = 0;
        vkEnumeratePhysicalDevices(bi, &n, nullptr);
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(bi, &n, pds.data());

        VkPhysicalDeviceVulkan13Features bf13 = {};
        bf13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        bf13.dynamicRendering = VK_TRUE;
        bf13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceFeatures2 bf2 = {};
        bf2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        bf2.pNext = &bf13;
        const float one = 1.0f;
        VkDeviceQueueCreateInfo bq = {};
        bq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        bq.queueFamilyIndex = c.family;
        bq.queueCount = 1;
        bq.pQueuePriorities = &one;
        VkDeviceCreateInfo bdci = {};
        bdci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        bdci.pNext = &bf2;
        bdci.queueCreateInfoCount = 1;
        bdci.pQueueCreateInfos = &bq;
        VkDevice bd = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDevice(pds[0], &bdci, nullptr, &bd));

        // The benchmark's own pointers, so that neither side touches the engine's table.
        auto d_begin = (PFN_vkBeginCommandBuffer)
            vkGetDeviceProcAddr(bd, "vkBeginCommandBuffer");
        auto d_end = (PFN_vkEndCommandBuffer)vkGetDeviceProcAddr(bd, "vkEndCommandBuffer");
        auto d_reset = (PFN_vkResetCommandBuffer)
            vkGetDeviceProcAddr(bd, "vkResetCommandBuffer");
        auto d_begin_rendering = (PFN_vkCmdBeginRendering)
            vkGetDeviceProcAddr(bd, "vkCmdBeginRendering");
        auto d_end_rendering = (PFN_vkCmdEndRendering)
            vkGetDeviceProcAddr(bd, "vkCmdEndRendering");
        auto d_bind = (PFN_vkCmdBindPipeline)vkGetDeviceProcAddr(bd, "vkCmdBindPipeline");
        auto d_draw = (PFN_vkCmdDraw)vkGetDeviceProcAddr(bd, "vkCmdDraw");

        BenchTarget bt = {};
        make_bench_target(bd, pds[0], c, bt);

        VkCommandPoolCreateInfo bcp = {};
        bcp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        bcp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        bcp.queueFamilyIndex = c.family;
        VkCommandPool bpool = VK_NULL_HANDLE;
        VK_CHECK((::vkCreateCommandPool)(bd, &bcp, nullptr, &bpool));
        VkCommandBufferAllocateInfo bcb = {};
        bcb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        bcb.commandPool = bpool;
        bcb.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        bcb.commandBufferCount = 1;
        VkCommandBuffer bench = VK_NULL_HANDLE;
        VK_CHECK((::vkAllocateCommandBuffers)(bd, &bcb, &bench));

        VkCommandBufferBeginInfo bb = {};
        bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VkRenderingAttachmentInfo batt = {};
        batt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        batt.imageView = bt.view;
        batt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        batt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        batt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo bri = {};
        bri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        bri.renderArea = { { 0, 0 }, { 64, 64 } };
        bri.layerCount = 1;
        bri.colorAttachmentCount = 1;
        bri.pColorAttachments = &batt;

        const int N = 200000;
        LARGE_INTEGER freq, a0, a1, b0, b1;
        QueryPerformanceFrequency(&freq);

        // ONE WARM PASS FIRST. The first touch of a command pool's memory is the allocator,
        // not the recording, and timing it once would put the whole cost on whichever side
        // happened to run first.
        for (int pass = 0; pass < 2; ++pass) {
            d_begin(bench, &bb);
            d_begin_rendering(bench, &bri);
            d_bind(bench, VK_PIPELINE_BIND_POINT_GRAPHICS, bt.pipeline);
            QueryPerformanceCounter(&a0);
            for (int i = 0; i < N; ++i) (::vkCmdDraw)(bench, 3, 1, 0, 0);
            QueryPerformanceCounter(&a1);
            d_end_rendering(bench);
            d_end(bench);
            d_reset(bench, 0);

            d_begin(bench, &bb);
            d_begin_rendering(bench, &bri);
            d_bind(bench, VK_PIPELINE_BIND_POINT_GRAPHICS, bt.pipeline);
            QueryPerformanceCounter(&b0);
            for (int i = 0; i < N; ++i) d_draw(bench, 3, 1, 0, 0);
            QueryPerformanceCounter(&b1);
            d_end_rendering(bench);
            d_end(bench);
            d_reset(bench, 0);
        }

        const double loader = 1e9 * double(a1.QuadPart - a0.QuadPart) / double(freq.QuadPart);
        const double direct = 1e9 * double(b1.QuadPart - b0.QuadPart) / double(freq.QuadPart);
        std::printf("bench    : no layers, %d valid draws, recorded twice\n", N);
        std::printf("bench    : through the loader's trampoline : %7.1f ns a call\n",
                    loader / N);
        std::printf("bench    : through the generated table     : %7.1f ns a call\n",
                    direct / N);
        std::printf("bench    : the trampoline costs %.2f ns a call, %.2f times\n",
                    (loader - direct) / N, loader / direct);

        (::vkDestroyCommandPool)(bd, bpool, nullptr);
        destroy_bench_target(bd, bt);
        (::vkDestroyDevice)(bd, nullptr);
        (::vkDestroyInstance)(bi, nullptr);
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
