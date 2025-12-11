#pragma once

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>

#ifdef OPTMATH_USE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace optmath {
namespace vulkan {

    /**
     * @brief Checks if Vulkan support is compiled in and available at runtime.
     */
    bool is_available();

    /**
     * @brief Singleton or Context manager for Vulkan.
     */
    class VulkanContext {
    public:
        static VulkanContext& get();

        bool init();
        void cleanup();

        // Very basic accessors for the demo
        // In real code, these would be encapsulated
#ifdef OPTMATH_USE_VULKAN
        VkDevice device = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t computeQueueFamilyIndex = 0;

        // Helper to find memory type
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
#endif
    private:
        VulkanContext() = default;
        ~VulkanContext() { cleanup(); }
        bool initialized = false;
#ifdef OPTMATH_USE_VULKAN
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
#endif
    };

    // --- Eigen Wrappers ---

    Eigen::VectorXf vulkan_vec_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf vulkan_vec_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    float           vulkan_vec_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    Eigen::VectorXf vulkan_conv1d(const Eigen::VectorXf& x, const Eigen::VectorXf& h);

}
}
