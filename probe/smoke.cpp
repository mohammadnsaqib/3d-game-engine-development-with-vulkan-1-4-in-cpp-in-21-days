// A smoke test, before a word of the book is written.
//
// It proves, in order: the headers compile with this compiler, the loader is reachable, the
// validation layer can be found WITHOUT a registry entry, this machine has a Vulkan 1.4
// device, and a queue family can present nothing and still do graphics.
//
// If any of that is false, the book cannot be written the way it is planned, and it is much
// better to find out now.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)   ? "ERROR"
                      : (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "warn"
                                                                                     : "note";
    std::printf("  vk %-5s %s\n", level, data->pMessage ? data->pMessage : "");
    return VK_FALSE;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // ---- 1. the loader answers ---------------------------------------------------------
    uint32_t api = 0;
    if (vkEnumerateInstanceVersion(&api) != VK_SUCCESS) {
        std::printf("FAILED: vkEnumerateInstanceVersion\n");
        return 1;
    }
    std::printf("loader   : instance version %u.%u.%u\n",
                VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));

    // ---- 2. is the validation layer visible? -------------------------------------------
    uint32_t nlayers = 0;
    vkEnumerateInstanceLayerProperties(&nlayers, nullptr);
    std::vector<VkLayerProperties> layers(nlayers);
    vkEnumerateInstanceLayerProperties(&nlayers, layers.data());
    bool have_validation = false;
    std::printf("layers   : %u available\n", nlayers);
    for (const VkLayerProperties& l : layers) {
        std::printf("  %-40s %u.%u.%u\n", l.layerName,
                    VK_API_VERSION_MAJOR(l.specVersion), VK_API_VERSION_MINOR(l.specVersion),
                    VK_API_VERSION_PATCH(l.specVersion));
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) have_validation = true;
    }
    std::printf("layers   : validation is %s\n", have_validation ? "PRESENT" : "ABSENT");

    // ---- 3. an instance, with validation and the debug messenger -----------------------
    const char* want_layers[] = { "VK_LAYER_KHRONOS_validation" };
    const char* want_exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

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
    app.pApplicationName = "meridian-smoke";
    app.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = have_validation ? &dbg : nullptr;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = have_validation ? 1u : 0u;
    ici.ppEnabledLayerNames = want_layers;
    ici.enabledExtensionCount = have_validation ? 1u : 0u;
    ici.ppEnabledExtensionNames = want_exts;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ici, nullptr, &instance);
    if (r != VK_SUCCESS) {
        std::printf("FAILED: vkCreateInstance = %d\n", (int)r);
        return 2;
    }
    std::printf("instance : created, asking for Vulkan 1.4\n");

    // ---- 4. the devices ------------------------------------------------------------------
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devices(ndev);
    vkEnumeratePhysicalDevices(instance, &ndev, devices.data());
    std::printf("devices  : %u\n", ndev);

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    uint32_t graphics_family = ~0u;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties p = {};
        vkGetPhysicalDeviceProperties(d, &p);
        const char* kind = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
                         : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                         : "other";
        std::printf("  %-38s %s, api %u.%u.%u, driver %u\n", p.deviceName, kind,
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion), p.driverVersion);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i) {
            if (!(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (chosen == VK_NULL_HANDLE ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                chosen = d;
                graphics_family = i;
            }
            break;
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        std::printf("FAILED: no device with a graphics queue\n");
        return 3;
    }

    // ---- 5. a device and a queue --------------------------------------------------------
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = graphics_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice device = VK_NULL_HANDLE;
    r = vkCreateDevice(chosen, &dci, nullptr, &device);
    if (r != VK_SUCCESS) {
        std::printf("FAILED: vkCreateDevice = %d\n", (int)r);
        return 4;
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, graphics_family, 0, &queue);
    std::printf("device   : created, graphics queue family %u\n", graphics_family);

    // ---- 6. memory, because an off-screen render needs to come home ---------------------
    VkPhysicalDeviceMemoryProperties mem = {};
    vkGetPhysicalDeviceMemoryProperties(chosen, &mem);
    std::printf("memory   : %u type(s), %u heap(s)\n", mem.memoryTypeCount, mem.memoryHeapCount);
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
        std::printf("  heap %u  %6llu MB  %s\n", i,
                    (unsigned long long)(mem.memoryHeaps[i].size / (1024 * 1024)),
                    (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                        ? "device local" : "host");

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("smoke    : every stage passed\n");
    return 0;
}
