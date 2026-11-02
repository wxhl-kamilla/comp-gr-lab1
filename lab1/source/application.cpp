#include "application.hpp"

#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace application {
namespace {

// Matrices are stored by columns, as GLSL expects. m[column * 4 + row].
struct Mat4 { float m[16]{}; };
Mat4 identity() { Mat4 a{}; for (int i = 0; i < 4; ++i) a.m[i * 5] = 1.f; return a; }
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 c{};
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                c.m[col * 4 + row] += a.m[k * 4 + row] * b.m[col * 4 + k];
    return c;
}
Mat4 translate(float x, float y, float z) {
    Mat4 a = identity(); a.m[12] = x; a.m[13] = y; a.m[14] = z; return a;
}
Mat4 scale(float x, float y, float z) {
    Mat4 a = identity(); a.m[0] = x; a.m[5] = y; a.m[10] = z; return a;
}
Mat4 rotateX(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m.m[5] = c; m.m[6] = s; m.m[9] = -s; m.m[10] = c; return m;
}
Mat4 rotateY(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m.m[0] = c; m.m[2] = -s; m.m[8] = s; m.m[10] = c; return m;
}
Mat4 rotateZ(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m.m[0] = c; m.m[1] = s; m.m[4] = -s; m.m[5] = c; return m;
}
constexpr float pi = 3.14159265358979323846f;
float radians(float degrees) { return degrees * pi / 180.f; }
Mat4 perspective(float aspect) {
    Mat4 a{};
    constexpr float near = 0.1f, far = 100.f;
    const float f = 1.f / std::tan(radians(55.f) / 2.f);
    a.m[0] = f / aspect; a.m[5] = f;
    a.m[10] = far / (near - far); a.m[11] = -1.f;
    a.m[14] = (far * near) / (near - far);
    return a; // Vulkan depth: 0..1.
}
Mat4 orthographic(float aspect) {
    const float h = 3.6f, w = h * aspect, near = 0.1f, far = 100.f;
    Mat4 a = identity();
    a.m[0] = 2.f / w; a.m[5] = 2.f / h;
    a.m[10] = 1.f / (near - far); a.m[14] = near / (near - far);
    return a;
}

struct Vertex { float position[3]; float color[3]; };
// Eight corners of a box with unequal dimensions; derive RGB from local XYZ.
const std::array<Vertex, 8> vertices = [] {
    std::array<Vertex, 8> result{};
    for (int i = 0; i < 8; ++i) {
        const float x = (i & 1) ? 0.8f : -0.8f;
        const float y = (i & 2) ? 0.5f : -0.5f;
        const float z = (i & 4) ? 0.4f : -0.4f;
        result[i] = {{x, y, z}, {0.25f + 0.75f * (x + 0.8f) / 1.6f,
                                  0.25f + 0.75f * (y + 0.5f) / 1.0f,
                                  0.25f + 0.75f * (z + 0.4f) / 0.8f}};
    }
    return result;
}();
constexpr uint16_t indices[] = {
    4,5,7, 4,7,6, 1,0,2, 1,2,3,
    0,4,6, 0,6,2, 5,1,3, 5,3,7,
    2,6,7, 2,7,3, 0,1,5, 0,5,4
};
struct alignas(16) Uniform { Mat4 mvp; float tint[4]; };
struct Buffer { VkBuffer handle = VK_NULL_HANDLE; VmaAllocation allocation = VK_NULL_HANDLE; };
Buffer vertexBuffer, indexBuffer;
std::array<Buffer, 2> uniformBuffers{};
VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
std::array<VkDescriptorSet, 2> sets{};
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

float position[3] = {0.f, 0.f, 0.f};
float rotation[3] = {18.f, 25.f, 0.f};
float size[3] = {1.f, 1.f, 1.f};
float tint[3] = {1.f, 1.f, 1.f};
float radius = 1.1f, speed = 1.f, phase = 0.f;
float secondOffset[3] = {2.5f, 0.f, -1.f};
bool paused = false, secondVisible = true, usePerspective = true;
double previousTime = -1.0;

bool createBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, Buffer& target) {
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = bytes, .usage = usage,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    const VmaAllocationCreateInfo memory{.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                         .usage = VMA_MEMORY_USAGE_AUTO};
    return vmaCreateBuffer(graphics::internal::context.allocator, &info, &memory,
                           &target.handle, &target.allocation, nullptr) == VK_SUCCESS;
}
void destroyBuffer(Buffer& b) {
    if (b.handle) vmaDestroyBuffer(graphics::internal::context.allocator, b.handle, b.allocation);
    b = {};
}
bool writeBuffer(const Buffer& b, const void* data, size_t bytes) {
    auto allocator = graphics::internal::context.allocator;
    void* mapped = nullptr;
    if (vmaMapMemory(allocator, b.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, data, bytes);
    vmaFlushAllocation(allocator, b.allocation, 0, bytes);
    vmaUnmapMemory(allocator, b.allocation);
    return true;
}
std::vector<char> load(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("Cannot open shader: ") + path);
    const auto count = file.tellg();
    if (count <= 0 || count % 4 != 0) throw std::runtime_error("Invalid SPIR-V file");
    std::vector<char> bytes(static_cast<size_t>(count));
    file.seekg(0); file.read(bytes.data(), count);
    if (!file) throw std::runtime_error("Cannot read SPIR-V file");
    return bytes;
}
VkShaderModule shader(const char* path) {
    auto bytes = load(path);
    VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = bytes.size(),
                                  .pCode = reinterpret_cast<const uint32_t*>(bytes.data())};
    VkShaderModule result = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS)
        throw std::runtime_error(std::string("Cannot create shader: ") + path);
    return result;
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;
    VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
    try { vert = shader(LAB_SHADER_DIR "box.vert.spv"); frag = shader(LAB_SHADER_DIR "box.frag.spv"); }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (vert) vkDestroyShaderModule(ctx.device, vert, nullptr);
        return false;
    }
    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage=VK_SHADER_STAGE_VERTEX_BIT, .module=vert, .pName="main"},
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage=VK_SHADER_STAGE_FRAGMENT_BIT, .module=frag, .pName="main"}
    };
    const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, position))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))}
    };
    const VkPipelineVertexInputStateCreateInfo input{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount=1, .pVertexBindingDescriptions=&binding,
        .vertexAttributeDescriptionCount=2, .pVertexAttributeDescriptions=attributes};
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    const VkPipelineViewportStateCreateInfo viewport{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount=1, .scissorCount=1};
    const VkPipelineRasterizationStateCreateInfo raster{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode=VK_POLYGON_MODE_FILL, .cullMode=VK_CULL_MODE_NONE,
        .frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth=1.f};
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
    const VkPipelineDepthStencilStateCreateInfo depth{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable=VK_TRUE, .depthWriteEnable=VK_TRUE,
        .depthCompareOp=VK_COMPARE_OP_LESS};
    const VkPipelineColorBlendAttachmentState blendAttachment{.colorWriteMask=
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount=1, .pAttachments=&blendAttachment};
    const VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount=2, .pDynamicStates=dynamicStates};
    const VkGraphicsPipelineCreateInfo info{
        .sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount=2, .pStages=stages, .pVertexInputState=&input,
        .pInputAssemblyState=&assembly, .pViewportState=&viewport,
        .pRasterizationState=&raster, .pMultisampleState=&multisample,
        .pDepthStencilState=&depth, .pColorBlendState=&blend,
        .pDynamicState=&dynamic, .layout=pipelineLayout,
        .renderPass=ctx.render_pass, .subpass=0};
    const auto result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    vkDestroyShaderModule(ctx.device, vert, nullptr);
    return result == VK_SUCCESS;
}
} // namespace

bool initialize() {
    auto& ctx = graphics::internal::context;
    if (!createBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer) ||
        !createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer) ||
        !writeBuffer(vertexBuffer, vertices.data(), sizeof(Vertex) * vertices.size()) ||
        !writeBuffer(indexBuffer, indices, sizeof(indices))) return (shutdown(), false);
    for (auto& b : uniformBuffers)
        if (!createBuffer(sizeof(Uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, b)) return (shutdown(), false);
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                 VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=1, .pBindings=&binding};
    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) return (shutdown(), false);
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1, .pSetLayouts=&setLayout};
    if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) return (shutdown(), false);
    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets=2, .poolSizeCount=1, .pPoolSizes=&poolSize};
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) return (shutdown(), false);
    const VkDescriptorSetLayout layouts[]{setLayout, setLayout};
    const VkDescriptorSetAllocateInfo allocateInfo{
        .sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=descriptorPool, .descriptorSetCount=2, .pSetLayouts=layouts};
    if (vkAllocateDescriptorSets(ctx.device, &allocateInfo, sets.data()) != VK_SUCCESS) return (shutdown(), false);
    for (size_t i = 0; i < sets.size(); ++i) {
        const VkDescriptorBufferInfo bufferInfo{uniformBuffers[i].handle, 0, sizeof(Uniform)};
        const VkWriteDescriptorSet write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=sets[i], .dstBinding=0, .descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo=&bufferInfo};
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }
    if (!createPipeline()) return (shutdown(), false);
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(ctx.device, setLayout, nullptr);
    for (auto& b : uniformBuffers) destroyBuffer(b);
    destroyBuffer(indexBuffer); destroyBuffer(vertexBuffer);
}

void update(double time) {
    if (previousTime < 0) previousTime = time;
    const double delta = std::clamp(time - previousTime, 0.0, 0.1);
    previousTime = time;
    if (!paused) phase += speed * static_cast<float>(delta);
    ImGui::Begin("Lab 1: parallelepiped (variant 3)");
    ImGui::Checkbox("Perspective projection", &usePerspective);
    ImGui::DragFloat3("Position", position, 0.02f);
    ImGui::DragFloat3("Rotation (degrees)", rotation, 0.5f);
    ImGui::DragFloat3("Scale", size, 0.01f, 0.1f, 4.f);
    ImGui::ColorEdit3("Color multiplier", tint);
    ImGui::Checkbox("Pause animation", &paused);
    ImGui::SliderFloat("Orbit speed", &speed, 0.f, 4.f);
    ImGui::SliderFloat("Orbit radius", &radius, 0.f, 2.f);
    ImGui::Checkbox("Second object", &secondVisible);
    if (secondVisible) ImGui::DragFloat3("Second object offset", secondOffset, 0.02f);
    ImGui::TextUnformatted("Vertex colors depend on local XYZ coordinates.");
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    if (!fd.command_buffer || !fd.framebuffer) return;
    const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                         static_cast<float>(ctx.swapchain_extent.height);
    const Mat4 projection = usePerspective ? perspective(aspect) : orthographic(aspect);
    const Mat4 view = translate(0.f, 0.f, -8.f); // Fixed camera at Z = +8.
    const Mat4 orbit = translate(radius * std::cos(phase),
                                 0.35f * std::sin(2.f * phase),
                                 radius * std::sin(phase));
    const Mat4 base = multiply(multiply(translate(position[0], position[1], position[2]), orbit),
        multiply(multiply(rotateZ(radians(rotation[2] + phase * 30.f)),
                          rotateY(radians(rotation[1] + phase * 50.f))),
                 multiply(rotateX(radians(rotation[0])), scale(size[0], size[1], size[2]))));
    const Mat4 models[]{base, multiply(translate(secondOffset[0], secondOffset[1], secondOffset[2]), base)};
    const size_t count = secondVisible ? 2 : 1;
    // prepare() waits for the previous GPU frame. Writing shared UBOs here is safe.
    for (size_t i = 0; i < count; ++i) {
        Uniform u{multiply(projection, multiply(view, models[i])),
                  {tint[0], tint[1], tint[2], 1.f}};
        if (!writeBuffer(uniformBuffers[i], &u, sizeof(u))) std::cerr << "Uniform update failed\n";
    }
    vkResetCommandBuffer(fd.command_buffer, 0);
    const VkCommandBufferBeginInfo begin{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(fd.command_buffer, &begin);
    const VkClearValue clear[]{
        {.color={{0.07f, 0.09f, 0.13f, 1.f}}},
        {.depthStencil={1.f, 0}}
    };
    const VkRenderPassBeginInfo pass{.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass=ctx.render_pass, .framebuffer=fd.framebuffer,
        .renderArea={{0,0},ctx.swapchain_extent}, .clearValueCount=2, .pClearValues=clear};
    vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkViewport vp{0.f, static_cast<float>(ctx.swapchain_extent.height),
        static_cast<float>(ctx.swapchain_extent.width),
        -static_cast<float>(ctx.swapchain_extent.height), 0.f, 1.f};
    const VkRect2D scissor{{0,0},ctx.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &vp);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer.handle, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT16);
    for (size_t i = 0; i < count; ++i) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &sets[i], 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}
} // namespace application
