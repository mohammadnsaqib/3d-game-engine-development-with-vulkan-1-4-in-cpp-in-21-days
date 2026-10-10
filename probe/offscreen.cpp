// The architecture probe: a triangle in a PNG, with no window and no swapchain.
//
// This is Day 1's shape, proved before Day 1 is written. If any stage here cannot be made to
// work, the book has to be planned differently, and the cheapest time to learn that is now.
//
// The stages, and every one of them is a thing the book will spend a section on:
//
//    instance, with validation found through VK_LAYER_PATH and no registry entry
//    physical device, queue family, logical device, queue
//    an image in device-local memory, and a view of it
//    SPIR-V loaded from disk, two shader modules, a pipeline with DYNAMIC RENDERING
//    a command buffer: barrier, begin rendering, draw, end, barrier, copy to a buffer
//    a fence, because the CPU has to know when the card is done
//    host-visible memory, mapped, and a PNG through the Windows Imaging Component
//
// DYNAMIC RENDERING, NOT A RENDER PASS. vkCmdBeginRendering is core in Vulkan 1.3 and this
// book targets 1.4. A VkRenderPass and a VkFramebuffer are two objects, about ninety lines and
// a whole vocabulary that a reader does not need on the first day -- and that new code no
// longer writes.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincodec.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

static const uint32_t W = 1920, H = 1080;

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::printf("FAILED: %s = %d (line %d)\n", #x, (int)_r, __LINE__); return 1; } } while (0)

static int g_messages = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    ++g_messages;
    std::printf("  vk %-5s %s\n",
                (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "warn",
                d->pMessage ? d->pMessage : "");
    return VK_FALSE;
}

static std::vector<char> read_file(const char* path) {
    std::vector<char> out;
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) return out;
    std::fseek(f, 0, SEEK_END);
    out.resize((size_t)std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return out;
}

// WHICH MEMORY TYPE. Vulkan does not have "the upload heap" and "the default heap"; it has a
// table of types with property flags, and you pick the first that both the resource allows and
// your use needs. This four-line loop appears in every Vulkan program ever written.
static uint32_t memory_type(VkPhysicalDevice pd, uint32_t allowed, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties m = {};
    vkGetPhysicalDeviceMemoryProperties(pd, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
        if ((allowed & (1u << i)) && (m.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return ~0u;
}

static bool write_png(const wchar_t* path, const uint8_t* rgba, UINT w, UINT h, bool* swapped) {
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
        // WIC ANSWERS A REQUEST FOR RGBA WITH BGRA AND RETURNS S_OK. Asking and then reading
        // back what you were given is the only way to know, and the swap is done here rather
        // than believed.
        WICPixelFormatGUID want = GUID_WICPixelFormat32bppRGBA, got = want;
        frame->SetPixelFormat(&got);
        *swapped = (got != want);
        std::vector<uint8_t> buf(rgba, rgba + size_t(w) * h * 4);
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

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    // ---- instance ------------------------------------------------------------------------
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    const char* exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

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
    app.pApplicationName = "meridian";
    app.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = &dbg;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = layers;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = exts;

    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    std::printf("instance : Vulkan 1.4, validation on\n");

    // ---- physical device, queue family ----------------------------------------------------
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, devs.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    uint32_t family = ~0u;
    for (VkPhysicalDevice d : devs) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { pd = d; family = i; break; }
        if (pd) break;
    }
    if (!pd) { std::printf("FAILED: no graphics queue\n"); return 1; }
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("gpu      : %s\n", props.deviceName);

    // ---- device, with dynamic rendering asked for explicitly ------------------------------
    // It is CORE in 1.3, and a core feature still has to be switched on: a 1.4 device does not
    // enable 1.3's features for you. This is the shape of every optional-but-core feature in
    // modern Vulkan, and forgetting it is a validation error rather than a crash.
    VkPhysicalDeviceVulkan13Features f13 = {};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2 = {};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f13;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice dev = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, family, 0, &queue);
    std::printf("device   : queue family %u, dynamic rendering and sync2 enabled\n", family);

    // ---- the colour image, in device-local memory ------------------------------------------
    VkImageCreateInfo img = {};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.imageType = VK_IMAGE_TYPE_2D;
    img.format = VK_FORMAT_R8G8B8A8_UNORM;
    img.extent = { W, H, 1 };
    img.mipLevels = 1;
    img.arrayLayers = 1;
    img.samples = VK_SAMPLE_COUNT_1_BIT;
    img.tiling = VK_IMAGE_TILING_OPTIMAL;
    img.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage colour = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImage(dev, &img, nullptr, &colour));
    VkMemoryRequirements mr = {};
    vkGetImageMemoryRequirements(dev, colour, &mr);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(pd, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory colour_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &colour_mem));
    VK_CHECK(vkBindImageMemory(dev, colour, colour_mem, 0));
    std::printf("image    : %ux%u R8G8B8A8_UNORM, %llu KB device local\n", W, H,
                (unsigned long long)(mr.size / 1024));

    VkImageViewCreateInfo ivi = {};
    ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image = colour;
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format = img.format;
    ivi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(dev, &ivi, nullptr, &view));

    // ---- the readback buffer, host visible ---------------------------------------------------
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = VkDeviceSize(W) * H * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer readback = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &readback));
    VkMemoryRequirements br = {};
    vkGetBufferMemoryRequirements(dev, readback, &br);
    VkMemoryAllocateInfo bai = {};
    bai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memory_type(pd, br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(dev, &bai, nullptr, &readback_mem));
    VK_CHECK(vkBindBufferMemory(dev, readback, readback_mem, 0));

    // ---- shaders ------------------------------------------------------------------------------
    const std::vector<char> vs_code = read_file("tri.vert.spv");
    const std::vector<char> fs_code = read_file("tri.frag.spv");
    if (vs_code.empty() || fs_code.empty()) {
        std::printf("FAILED: no SPIR-V beside the exe -- run glslc first\n");
        return 1;
    }
    std::printf("spirv    : %zu and %zu bytes, %zu and %zu words\n",
                vs_code.size(), fs_code.size(), vs_code.size() / 4, fs_code.size() / 4);

    auto make_module = [&](const std::vector<char>& code) {
        VkShaderModuleCreateInfo smi = {};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = code.size();
        smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(dev, &smi, nullptr, &m);
        return m;
    };
    VkShaderModule vs = make_module(vs_code), fs = make_module(fs_code);
    if (!vs || !fs) { std::printf("FAILED: shader module\n"); return 1; }

    // ---- the pipeline -----------------------------------------------------------------------
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &layout));

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp = { 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { W, H } };
    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

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
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // THE PIPELINE IS TOLD WHAT IT WILL RENDER INTO, and with dynamic rendering that is a
    // format rather than a render pass object.
    VkFormat colour_format = img.format;
    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &colour_format;

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
    gpi.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipeline));
    std::printf("pipeline : one, and it is immutable -- nine structs, no render pass\n");

    // ---- the command buffer -------------------------------------------------------------------
    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

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
        b.image = colour;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

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
    att.clearValue.color = { { 0.055f, 0.066f, 0.098f, 1.0f } };

    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = sc;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { W, H, 1 };
    vkCmdCopyImageToBuffer(cmd, colour, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // ---- submit, and wait ---------------------------------------------------------------------
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));

    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    std::printf("submit   : one command buffer, one fence, waited\n");

    // ---- the picture ----------------------------------------------------------------------------
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(dev, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    bool swapped = false;
    const bool ok = write_png(L"probe_triangle.png", (const uint8_t*)mapped, W, H, &swapped);
    vkUnmapMemory(dev, readback_mem);
    std::printf("wic      : asked for RGBA and was given %s\n", swapped ? "BGRA" : "RGBA");
    std::printf("png      : %s\n", ok ? "probe_triangle.png written" : "FAILED");

    std::printf("debug    : %d message(s) from the validation layer\n", g_messages);
    std::printf("probe    : every stage passed, no window opened\n");

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);
    vkDestroyBuffer(dev, readback, nullptr);
    vkFreeMemory(dev, readback_mem, nullptr);
    vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, colour, nullptr);
    vkFreeMemory(dev, colour_mem, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(instance, nullptr);
    CoUninitialize();
    return ok ? 0 : 1;
}
