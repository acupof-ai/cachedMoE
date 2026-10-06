#include "gpu/vulkan/descriptor.h"

#include <format>

namespace cachedmoe::gpu {

DescriptorPool::~DescriptorPool() { destroy(); }

#if !defined(CACHEDMOE_ENABLE_VULKAN)

Result<void> DescriptorPool::create(Device&, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "built without CACHEDMOE_ENABLE_VULKAN");
}
void DescriptorPool::destroy() { device_ = nullptr; }
void DescriptorPool::reset() {}

#else

Result<void> DescriptorPool::create(Device& device, uint32_t max_sets,
                                    uint32_t max_storage_buffers) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    device_ = &device;
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_storage_buffers};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = max_sets;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &size;
    if (VkResult r = vkCreateDescriptorPool(device.handle(), &ci, nullptr, &pool_); r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkCreateDescriptorPool failed ({})", static_cast<int>(r)));
    return {};
}

void DescriptorPool::destroy() {
    if (pool_ && device_ && device_->valid())
        vkDestroyDescriptorPool(device_->handle(), pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    device_ = nullptr;
}

void DescriptorPool::reset() {
    if (pool_ && device_ && device_->valid())
        vkResetDescriptorPool(device_->handle(), pool_, 0);
}

Result<VkDescriptorSet> DescriptorPool::allocate(const Pipeline& pipeline,
                                                 const std::vector<BufferBinding>& bindings) {
    if (!pool_) return fail(Err::FailedPrecondition, "descriptor pool is not created");
    if (!pipeline.valid()) return fail(Err::FailedPrecondition, "pipeline is not created");

    VkDescriptorSetLayout layout = pipeline.set_layout();
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (VkResult r = vkAllocateDescriptorSets(device_->handle(), &ai, &set); r != VK_SUCCESS)
        return fail(Err::ResourceExhausted,
                    std::format("vkAllocateDescriptorSets failed ({})", static_cast<int>(r)));

    std::vector<VkDescriptorBufferInfo> infos;
    std::vector<VkWriteDescriptorSet>   writes;
    infos.reserve(bindings.size());
    writes.reserve(bindings.size());
    for (const BufferBinding& b : bindings) {
        if (b.buffer == VK_NULL_HANDLE) continue;
        infos.push_back(VkDescriptorBufferInfo{b.buffer, b.offset, b.range ? b.range : VK_WHOLE_SIZE});
    }
    size_t i = 0;
    for (const BufferBinding& b : bindings) {
        if (b.buffer == VK_NULL_HANDLE) continue;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = b.binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &infos[i++];
        writes.push_back(w);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(device_->handle(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    return set;
}

#endif  // CACHEDMOE_ENABLE_VULKAN

}  // namespace cachedmoe::gpu

#include <cstring>
namespace cachedmoe::gpu {
ArgumentPages::~ArgumentPages(){destroy();}
void ArgumentPages::destroy(){descriptors_.destroy();if(allocator_&&buffer_.valid())allocator_->free(buffer_);allocator_=nullptr;buffer_={};used_=capacity_=0;
#if defined(CACHEDMOE_ENABLE_VULKAN)
sets_.clear();
#endif
}
Result<void> ArgumentPages::create(Device& dev,MemoryAllocator& alloc,uint32_t capacity){
#if defined(CACHEDMOE_ENABLE_VULKAN)
    if(valid())return fail(Err::AlreadyExists,"argument pages already created");
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(dev.physical(),&props);
    stride_=std::max<uint64_t>(256,props.limits.minStorageBufferOffsetAlignment);
    allocator_=&alloc;capacity_=capacity;
    auto b=alloc.allocate(capacity*stride_,true,false);if(!b)return std::unexpected(b.error());buffer_=*b;
    sets_.resize(capacity,VK_NULL_HANDLE);return descriptors_.create(dev,capacity,capacity);
#else
    return fail(Err::Unavailable,"no Vulkan");
#endif
}
#if defined(CACHEDMOE_ENABLE_VULKAN)
Result<VkDescriptorSet> ArgumentPages::snapshot(const Pipeline& pipe,const uint64_t* ptr){
    if(!valid()||used_>=capacity_)return fail(Err::ResourceExhausted,"immutable argument pages exhausted");
    const auto i=used_++;std::memcpy(static_cast<std::byte*>(buffer_.host_ptr)+i*stride_,ptr,256);
    if(!sets_[i]){auto s=descriptors_.allocate(pipe,{{0,i*stride_,256,buffer_.buffer}});
        if(!s)return std::unexpected(s.error());sets_[i]=*s;}
    return sets_[i];
}
#endif
}
