#include "application.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <functional>
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace application {
    namespace {
        // Матрицы хранятся по столбцам: m[номер столбца * 4 + номер строки].
        struct Mat4 { float m[16]{}; };
        Mat4 identity() { Mat4 a{}; for (int i = 0; i < 4; ++i) a.m[i * 5] = 1.f; return a; }
        // multiply(A, B) применяет к точке сначала B, затем A.
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
        // Угол fov сужает или расширяет обзор; aspect учитывает пропорции окна.
        Mat4 perspective(float aspect, float fov) {
            Mat4 a{};
            constexpr float near = 0.1f, far = 100.f;
            const float f = 1.f / std::tan(radians(fov) / 2.f);
            a.m[0] = f / aspect; a.m[5] = f;
            a.m[10] = far / (near - far); a.m[11] = -1.f;
            a.m[14] = (far * near) / (near - far);
            return a; // Глубина Vulkan находится в диапазоне от 0 до 1.
        }
        Mat4 orthographic(float aspect) {
            const float h = 3.6f, w = h * aspect, near = 0.1f, far = 100.f;
            Mat4 a = identity();
            a.m[0] = 2.f / w; a.m[5] = 2.f / h;
            a.m[10] = 1.f / (near - far); a.m[14] = near / (near - far);
            return a;
        }


        struct Vec3 { float x, y, z; };
        Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
        Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
        Vec3 operator*(Vec3 a, float k) { return { a.x * k, a.y * k, a.z * k }; }
        float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        Vec3 cross(Vec3 a, Vec3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
        Vec3 normalize(Vec3 a) {
            float length = std::sqrt(dot(a, a));
            return length > 0.00001f ? a * (1.f / length) : Vec3{ 0,-1,0 };
        }
        struct Vertex { Vec3 position; Vec3 normal; Vec3 color; float uv[2]{}; };
        std::vector<uint16_t> indices;
        std::vector<Vertex> vertices;

        // У каждой грани свои вершины: в одном углу коробки встречаются три разные нормали.
        void makeBox() {
            std::array<Vec3, 8> corners{};
            for (int i = 0; i < 8; ++i)
                corners[i] = { (i & 1) ? 0.8f : -0.8f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.4f : -0.4f };
            const int quads[6][4] = { {4,5,7,6},{1,0,2,3},{0,4,6,2},
                                      {5,1,3,7},{2,6,7,3},{0,1,5,4} };
            const float uv[4][2] = {{0,1},{1,1},{1,0},{0,0}};
            for (auto& face : quads) {
                Vec3 a = corners[face[0]], b = corners[face[1]], c = corners[face[2]];
                Vec3 n = normalize(cross(b-a,c-a));
                auto base = static_cast<uint16_t>(vertices.size());
                for (int j = 0; j < 4; ++j) {
                    Vec3 v=corners[face[j]];
                    vertices.push_back({v,n,{0.25f+0.75f*(v.x+0.8f)/1.6f,
                        0.25f+0.75f*(v.y+0.5f),0.25f+0.75f*(v.z+0.4f)/0.8f},{uv[j][0],uv[j][1]}});
                }
                for (int index : {0,1,2,0,2,3}) indices.push_back(base + index);
            }
        }

        // Используем блоки по 16 байт, чтобы размещение C++ совпало с std140/std430 в GLSL.
        struct alignas(16) Vec4 { float x = 0, y = 0, z = 0, w = 0; };
        struct alignas(16) SceneUniform {
            Mat4 viewProjection;
            Vec4 cameraPosition, ambient, direction, directionalColor;
            std::array<int32_t, 4> counts{};
        };
        struct alignas(16) ModelUniform {
            Mat4 model, normalMatrix;
            Vec4 albedo, specular, parameters;
        };
        struct alignas(16) PointLight { Vec4 position, color; };
        struct alignas(16) SpotLight { Vec4 position, direction, color, angles; };
        static_assert(sizeof(SceneUniform) == 144 && offsetof(SceneUniform, counts) == 128);
        static_assert(sizeof(ModelUniform) == 176 && offsetof(ModelUniform, albedo) == 128);
        static_assert(sizeof(PointLight) == 32 && sizeof(SpotLight) == 64);
        struct Buffer {
            VkBuffer handle = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
        };
        Buffer vertexBuffer, sceneBuffer, pointBuffer, spotBuffer, lineBuffer;
        Buffer modelBuffer, indexBuffer;
        VkDeviceSize modelStride = 0;
        size_t modelCapacity = 2;
        VkCommandPool uploadPool = VK_NULL_HANDLE;
        struct Texture { VkImage image{}; VmaAllocation allocation{}; VkImageView view{}; };
        std::array<Texture, 2> textures{};
        VkSampler sampler{};
        std::array<VkDescriptorSet, 2> materialSets{};
        int textureChoice[2] = {0,1};
        bool samplingEffect[2] = {false,true};
        float effectTime = 0, effectSpeed = 1, effectStrength = 0.4f;
        bool animateTexture = true;
        size_t pointCapacity = 0, spotCapacity = 0, lineCapacity = 0;
        VkDescriptorSetLayout sceneLayout = VK_NULL_HANDLE, modelLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet sceneSet = VK_NULL_HANDLE;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE, linePipeline = VK_NULL_HANDLE;

        float position[3] = { 0,0,0 }, rotation[3] = { 18,25,0 }, size[3] = { 1,1,1 };
        float secondOffset[3] = { 2.5f,0,-1 };
        float radius = 1.1f, speed = 1, phase = 0;
        bool paused = true, secondVisible = true, usePerspective = true, vertexColors = false;
        float cameraPosition[3] = { 0,1.5f,9 }, cameraRotation[3] = { -9,0,0 }, fov = 55;
        float ambientColor[3] = { 0.7f,0.8f,1 }, ambientIntensity = 0.12f;
        float direction[3] = { -0.5f,-1,-0.3f }, directionalColor[3] = { 1,0.94f,0.85f };
        float directionalIntensity = 0.7f;
        bool directionalEnabled = true, showLights = true, showCones = true;
        float coneLength = 2.5f;
        struct Material {
            float albedo[3] = { 1,1,1 };
            float specular[3] = { 1,1,1 };
            float shininess = 64;
        };
        std::array<Material, 2> materials{};
        std::vector<PointLight> pointLights;
        std::vector<SpotLight> spotLights;
        double previousTime = -1;
        struct Marker { Mat4 model; Vec4 color; };
        std::vector<Marker> markers;
        std::vector<Vertex> coneVertices;

        // usage задаёт назначение буфера, bytes — размер памяти в байтах.
        bool createBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, Buffer& target) {
            const VkBufferCreateInfo info{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                          .size = bytes, .usage = usage,
                                          .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
            const VmaAllocationCreateInfo memory{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                                 .usage = VMA_MEMORY_USAGE_AUTO };
            return vmaCreateBuffer(graphics::internal::context.allocator, &info, &memory,
                &target.handle, &target.allocation, nullptr) == VK_SUCCESS;
        }
        void destroyBuffer(Buffer& b) {
            if (b.handle) vmaDestroyBuffer(graphics::internal::context.allocator, b.handle, b.allocation);
            b = {};
        }
        // Копируем данные CPU в память GPU и сбрасываем кэш для видимости записи.
        bool writeBuffer(const Buffer& b, const void* data, size_t bytes, VkDeviceSize offset = 0) {
            auto allocator = graphics::internal::context.allocator;
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, b.allocation, &mapped) != VK_SUCCESS) return false;
            std::memcpy(static_cast<char*>(mapped) + offset, data, bytes);
            vmaFlushAllocation(allocator, b.allocation, offset, bytes);
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
            VkShaderModuleCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize = bytes.size(),
                                          .pCode = reinterpret_cast<const uint32_t*>(bytes.data()) };
            VkShaderModule result = VK_NULL_HANDLE;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS)
                throw std::runtime_error(std::string("Cannot create shader: ") + path);
            return result;
        }


        // Исключение останавливает кадр при ошибке Vulkan вместо работы с пустым ресурсом.
        void require(bool success, const char* message) {
            if (!success) throw std::runtime_error(message);
        }
        // Короткая команда копирования. Ожидание гарантирует, что staging можно освободить.
        void upload(const std::function<void(VkCommandBuffer)>& record) {
            auto& ctx = graphics::internal::context;
            VkCommandBuffer command{};
            const VkCommandBufferAllocateInfo allocation{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool=uploadPool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
            require(vkAllocateCommandBuffers(ctx.device,&allocation,&command)==VK_SUCCESS,"Upload command");
            const VkCommandBufferBeginInfo begin{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            require(vkBeginCommandBuffer(command,&begin)==VK_SUCCESS,"Upload begin");
            record(command);
            require(vkEndCommandBuffer(command)==VK_SUCCESS,"Upload end");
            const VkSubmitInfo submit{.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&command};
            require(vkQueueSubmit(ctx.graphics_queue,1,&submit,VK_NULL_HANDLE)==VK_SUCCESS,"Upload submit");
            require(vkQueueWaitIdle(ctx.graphics_queue)==VK_SUCCESS,"Upload wait");
            vkFreeCommandBuffers(ctx.device,uploadPool,1,&command);
        }
        // Только staging доступен CPU; целевой буфер требует DEVICE_LOCAL память.
        void uploadGpu(Buffer& target, const void* data, VkDeviceSize bytes, VkBufferUsageFlags usage) {
            auto& ctx = graphics::internal::context;
            Buffer staging{};
            require(createBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging),"Staging buffer");
            require(writeBuffer(staging,data,static_cast<size_t>(bytes)),"Staging write");
            if (!target.handle) {
                const VkBufferCreateInfo info{.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,
                    .usage=usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
                const VmaAllocationCreateInfo memory{.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    .requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
                require(vmaCreateBuffer(ctx.allocator,&info,&memory,&target.handle,&target.allocation,nullptr)==VK_SUCCESS,"GPU buffer");
            }
            upload([&](VkCommandBuffer command) {
                const VkBufferCopy region{0,0,bytes};
                vkCmdCopyBuffer(command,staging.handle,target.handle,1,&region);
                const VkBufferMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask=VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|VK_ACCESS_INDEX_READ_BIT,
                    .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                    .buffer=target.handle,.offset=0,.size=bytes};
                vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    0,0,nullptr,1,&barrier,0,nullptr);
            });
            destroyBuffer(staging);
        }
        void loadTexture(Texture& texture, const char* filename, bool srgb) {
            auto& ctx = graphics::internal::context;
            int width=0,height=0,channels=0;
            std::string path=std::string(LAB_TEXTURE_DIR)+filename;
            stbi_uc* pixels=stbi_load(path.c_str(),&width,&height,&channels,4);
            require(pixels!=nullptr,("Cannot load texture: "+path).c_str());
            VkDeviceSize bytes=static_cast<VkDeviceSize>(width)*height*4;
            Buffer staging{};
            require(createBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging),"Texture staging");
            bool written=writeBuffer(staging,pixels,static_cast<size_t>(bytes));
            stbi_image_free(pixels);
            require(written,"Texture upload");
            VkFormat format=srgb?VK_FORMAT_R8G8B8A8_SRGB:VK_FORMAT_R8G8B8A8_UNORM;
            const VkImageCreateInfo info{.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,
                .format=format,.extent={static_cast<uint32_t>(width),static_cast<uint32_t>(height),1},
                .mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
                .usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
            const VmaAllocationCreateInfo memory{.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
            require(vmaCreateImage(ctx.allocator,&info,&memory,&texture.image,&texture.allocation,nullptr)==VK_SUCCESS,"Texture image");
            upload([&](VkCommandBuffer command) {
                VkImageMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask=0,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                    .image=texture.image,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
                vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
                const VkBufferImageCopy copy{.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
                    .imageExtent={static_cast<uint32_t>(width),static_cast<uint32_t>(height),1}};
                vkCmdCopyBufferToImage(command,staging.handle,texture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
                barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
                barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
            });
            destroyBuffer(staging);
            const VkImageViewCreateInfo view{.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=texture.image,
                .viewType=VK_IMAGE_VIEW_TYPE_2D,.format=format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
            require(vkCreateImageView(ctx.device,&view,nullptr,&texture.view)==VK_SUCCESS,"Texture view");
        }
        void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
            const Buffer& buffer, VkDeviceSize bytes) {
            VkDescriptorBufferInfo info{ buffer.handle,0,bytes };
            VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,.dstBinding = binding,.descriptorCount = 1,
                .descriptorType = type,.pBufferInfo = &info };
            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
        }
        // В том же массиве UBO находятся и обычные модели, и маркеры света, и линии.
        void ensureModels(size_t count) {
            if(count<=modelCapacity) return;
            require(count<=UINT32_MAX/modelStride,"Dynamic UBO offset exceeds uint32");
            size_t next=std::max(count,modelCapacity*2);
            next=std::min(next,static_cast<size_t>(UINT32_MAX/modelStride));
            Buffer replacement{};
            require(createBuffer(modelStride*next,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,replacement),"Grow dynamic UBO");
            destroyBuffer(modelBuffer); modelBuffer=replacement; modelCapacity=next;
            for(auto set:materialSets)
                bindBuffer(set,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,modelBuffer,sizeof(ModelUniform));
        }
        // Вызывается после prepare(): предыдущий кадр уже закончен, буфер можно заменить.
        // Вместимость растёт по необходимости; в шейдере нет фиксированного лимита света.
        void ensureStorage(Buffer& buffer, size_t& capacity, size_t count, size_t stride,
            uint32_t binding) {
            if (capacity >= std::max(size_t(1), count)) return;
            size_t next = std::max({ size_t(1),count,capacity * 2 });
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
            require(next <= properties.limits.maxStorageBufferRange / stride, "Storage buffer exceeds GPU limit");
            Buffer replacement{};
            require(createBuffer(next * stride, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, replacement),
                "Cannot allocate storage buffer");
            destroyBuffer(buffer); buffer = replacement; capacity = next;
            bindBuffer(sceneSet, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer, next * stride);
        }
        // Вершины и шейдеры одинаковые; для конусов меняем треугольники на линии.
        bool createPipeline(bool lines, VkPipeline& target) {
            auto& ctx = graphics::internal::context;
            VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
            try { vert = shader(LAB_SHADER_DIR "box.vert.spv"); frag = shader(LAB_SHADER_DIR "box.frag.spv"); }
            catch (const std::exception& e) {
                std::cerr << e.what() << '\n';
                if (vert) vkDestroyShaderModule(ctx.device, vert, nullptr);
                return false;
            }
            const VkPipelineShaderStageCreateInfo stages[] = {
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"},
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"}
            };
            const VkVertexInputBindingDescription binding{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            const VkVertexInputAttributeDescription attributes[] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, position))},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, normal))},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv))}
            };
            const VkPipelineVertexInputStateCreateInfo input{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
                .vertexAttributeDescriptionCount = 4, .pVertexAttributeDescriptions = attributes };
            const VkPipelineInputAssemblyStateCreateInfo assembly{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = lines ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
            const VkPipelineViewportStateCreateInfo viewport{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1, .scissorCount = 1 };
            const VkPipelineRasterizationStateCreateInfo raster{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.f };
            const VkPipelineMultisampleStateCreateInfo multisample{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
            const VkPipelineDepthStencilStateCreateInfo depth{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS };
            const VkPipelineColorBlendAttachmentState blendAttachment{ .colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
            const VkPipelineColorBlendStateCreateInfo blend{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1, .pAttachments = &blendAttachment };
            const VkDynamicState dynamicStates[]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            const VkPipelineDynamicStateCreateInfo dynamic{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2, .pDynamicStates = dynamicStates };
            const VkGraphicsPipelineCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2, .pStages = stages, .pVertexInputState = &input,
                .pInputAssemblyState = &assembly, .pViewportState = &viewport,
                .pRasterizationState = &raster, .pMultisampleState = &multisample,
                .pDepthStencilState = &depth, .pColorBlendState = &blend,
                .pDynamicState = &dynamic, .layout = pipelineLayout,
                .renderPass = ctx.render_pass, .subpass = 0 };
            const auto result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &target);
            vkDestroyShaderModule(ctx.device, frag, nullptr);
            vkDestroyShaderModule(ctx.device, vert, nullptr);
            return result == VK_SUCCESS;
        }

        // Матрица камеры: T * Ry * Rx * Rz. Вид — её обратная: Rz(-z)*Rx(-x)*Ry(-y)*T(-p).
        Mat4 cameraView() {
            Mat4 inverseRotation = multiply(rotateZ(radians(-cameraRotation[2])),
                multiply(rotateX(radians(-cameraRotation[0])), rotateY(radians(-cameraRotation[1]))));
            return multiply(inverseRotation, translate(-cameraPosition[0], -cameraPosition[1], -cameraPosition[2]));
        }
        Mat4 objectRotation() {
            return multiply(rotateZ(radians(rotation[2] + phase * 30)),
                multiply(rotateY(radians(rotation[1] + phase * 50)), rotateX(radians(rotation[0]))));
        }

        // Белые линии показывают ось и оба конуса: внутренний и внешний углы — полууглы.
        void makeCone(const SpotLight& light) {
            Vec3 p{ light.position.x,light.position.y,light.position.z };
            Vec3 d = normalize({ light.direction.x,light.direction.y,light.direction.z });
            Vec3 helper = std::abs(d.y) < 0.95f ? Vec3{ 0,1,0 } : Vec3{ 1,0,0 };
            Vec3 u = normalize(cross(d, helper)), v = cross(d, u);
            auto line = [](Vec3 a, Vec3 b) {
                coneVertices.push_back({ a,{0,1,0},{1,1,1} });
                coneVertices.push_back({ b,{0,1,0},{1,1,1} });
                };
            line(p, p + d * coneLength);
            for (float angle : {light.angles.x, light.angles.y}) {
                float r = coneLength * std::tan(radians(angle));
                Vec3 center = p + d * coneLength;
                for (int i = 0; i < 32; ++i) {
                    float a = 2 * pi * i / 32, b = 2 * pi * (i + 1) / 32;
                    Vec3 q = center + (u * std::cos(a) + v * std::sin(a)) * r;
                    Vec3 next = center + (u * std::cos(b) + v * std::sin(b)) * r;
                    line(q, next);
                    if (i % 8 == 0) line(p, q);
                }
            }
        }
    } // Конец пространства имён.

    bool initialize() {
        auto& ctx = graphics::internal::context;
        try {
            makeBox();
            const VkCommandPoolCreateInfo pool{.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,.queueFamilyIndex=ctx.graphics_queue_index};
            require(vkCreateCommandPool(ctx.device,&pool,nullptr,&uploadPool)==VK_SUCCESS,"Upload pool");
            uploadGpu(vertexBuffer,vertices.data(),vertices.size()*sizeof(Vertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            uploadGpu(indexBuffer,indices.data(),indices.size()*sizeof(uint16_t),VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            require(createBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer), "Scene buffer");
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(ctx.physical_device,&properties);
            VkDeviceSize alignment=properties.limits.minUniformBufferOffsetAlignment;
            modelStride=(sizeof(ModelUniform)+alignment-1)/alignment*alignment;
            require(createBuffer(modelStride*2,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,modelBuffer),"Dynamic models");
            // Реальные фотографии тканей из открытой коллекции Met Museum.
            const char* files[]={"dragons_clouds.jpg","dragon_medallions.jpg"};
            for (size_t i=0;i<textures.size();++i) loadTexture(textures[i],files[i],true);
            const VkSamplerCreateInfo samplerInfo{.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter=VK_FILTER_LINEAR,.minFilter=VK_FILTER_LINEAR,.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,.addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT,.maxLod=0};
            require(vkCreateSampler(ctx.device,&samplerInfo,nullptr,&sampler)==VK_SUCCESS,"Texture sampler");
            // set=0: общая сцена, массив точечных источников и массив прожекторов.
            const VkDescriptorSetLayoutBinding sceneBindings[] = {
                {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr} };
            VkDescriptorSetLayoutCreateInfo layout{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,.pBindings = sceneBindings };
            require(vkCreateDescriptorSetLayout(ctx.device, &layout, nullptr, &sceneLayout) == VK_SUCCESS, "Scene layout");
            // Один набор на текстуру: общий dynamic UBO и изображение-сэмплер.
            const VkDescriptorSetLayoutBinding modelBindings[] = {
                {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
            layout.bindingCount=2; layout.pBindings=modelBindings;
            require(vkCreateDescriptorSetLayout(ctx.device,&layout,nullptr,&modelLayout)==VK_SUCCESS,"Material layout");
            const VkDescriptorSetLayout layouts[] = { sceneLayout,modelLayout };
            const VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,0,16 };
            const VkPipelineLayoutCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,.pSetLayouts = layouts,.pushConstantRangeCount = 1,.pPushConstantRanges = &push };
            require(vkCreatePipelineLayout(ctx.device, &pipelineInfo, nullptr, &pipelineLayout) == VK_SUCCESS, "Pipeline layout");
            const VkDescriptorPoolSize poolSizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2}};
            const VkDescriptorPoolCreateInfo poolInfo{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets=3,.poolSizeCount=4,.pPoolSizes=poolSizes};
            require(vkCreateDescriptorPool(ctx.device,&poolInfo,nullptr,&descriptorPool)==VK_SUCCESS,"Descriptor pool");
            const VkDescriptorSetLayout allocatedLayouts[]={sceneLayout,modelLayout,modelLayout};
            const VkDescriptorSetAllocateInfo allocation{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool=descriptorPool,.descriptorSetCount=3,.pSetLayouts=allocatedLayouts};
            VkDescriptorSet allocated[3]{};
            require(vkAllocateDescriptorSets(ctx.device,&allocation,allocated)==VK_SUCCESS,"Descriptor sets");
            sceneSet=allocated[0];
            bindBuffer(sceneSet,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,sceneBuffer,sizeof(SceneUniform));
            for (size_t i=0;i<materialSets.size();++i) {
                materialSets[i]=allocated[i+1];
                bindBuffer(materialSets[i],0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,modelBuffer,sizeof(ModelUniform));
                VkDescriptorImageInfo image{sampler,textures[i].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=materialSets[i],
                    .dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&image};
                vkUpdateDescriptorSets(ctx.device,1,&write,0,nullptr);
            }
            // Даже при нулевом числе источников дескриптор должен указывать на настоящий буфер.
            ensureStorage(pointBuffer, pointCapacity, 1, sizeof(PointLight), 1);
            ensureStorage(spotBuffer, spotCapacity, 1, sizeof(SpotLight), 2);
            require(createPipeline(false, pipeline), "Triangle pipeline");
            require(createPipeline(true, linePipeline), "Line pipeline");
            pointLights.push_back({ {-0.5f,2.2f,2,0},{0.25f,0.55f,1,9} });
            spotLights.push_back({ {3,3,3,0},{-0.6f,-0.6f,-1,0},{1,0.4f,0.15f,20},{15,25,0,0} });
            auto& style=ImGui::GetStyle();
            style.WindowRounding=4; style.FrameRounding=3;
            style.Colors[ImGuiCol_WindowBg]=ImVec4(0.055f,0.075f,0.07f,0.97f);
            style.Colors[ImGuiCol_Text]=ImVec4(0.91f,0.86f,0.72f,1);
            style.Colors[ImGuiCol_Header]=ImVec4(0.22f,0.38f,0.31f,1);
            style.Colors[ImGuiCol_Button]=ImVec4(0.42f,0.12f,0.13f,1);
            style.Colors[ImGuiCol_FrameBg]=ImVec4(0.13f,0.2f,0.18f,1);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << '\n'; shutdown(); return false;
        }
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);
        if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
        if (linePipeline) vkDestroyPipeline(ctx.device, linePipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
        if (sceneLayout) vkDestroyDescriptorSetLayout(ctx.device, sceneLayout, nullptr);
        if (modelLayout) vkDestroyDescriptorSetLayout(ctx.device, modelLayout, nullptr);
        for (auto& texture : textures) {
            if(texture.view) vkDestroyImageView(ctx.device,texture.view,nullptr);
            if(texture.image) vmaDestroyImage(ctx.allocator,texture.image,texture.allocation);
            texture={};
        }
        if(sampler) vkDestroySampler(ctx.device,sampler,nullptr);
        if(uploadPool) vkDestroyCommandPool(ctx.device,uploadPool,nullptr);
        sampler=VK_NULL_HANDLE; uploadPool=VK_NULL_HANDLE;
        destroyBuffer(modelBuffer); destroyBuffer(indexBuffer);
        for (Buffer* b : { &vertexBuffer,&sceneBuffer,&pointBuffer,&spotBuffer,&lineBuffer }) destroyBuffer(*b);
        pipeline = linePipeline = VK_NULL_HANDLE; pipelineLayout = VK_NULL_HANDLE;
        descriptorPool = VK_NULL_HANDLE; sceneLayout = modelLayout = VK_NULL_HANDLE;
        pointCapacity = spotCapacity = lineCapacity = 0; modelCapacity=2;
        vertices.clear(); indices.clear(); pointLights.clear(); spotLights.clear(); previousTime = -1;
    }

    void update(double time) {
        if(animateTexture && previousTime>=0) effectTime+=static_cast<float>(time-previousTime)*effectSpeed;
        if (previousTime < 0) previousTime = time;
        float delta = static_cast<float>(std::clamp(time - previousTime, 0.0, 0.1));
        previousTime = time;
        if (!paused) phase += speed * delta;
        ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(470, 690), ImGuiCond_FirstUseEver);
        ImGui::Begin("Lab 3: texture mapping - variant 3");
        ImGui::TextColored(ImVec4(0.8f,0.64f,0.35f,1),"SILK PAVILION / TEXTURE MAPPING");
        ImGui::Separator();
        ImGui::PushItemWidth(225);
        // ImGui сразу меняет эти переменные. Матрицы и буферы обновятся в render этого кадра.
        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Checkbox("Perspective", &usePerspective);
            ImGui::DragFloat3("Camera position", cameraPosition, 0.05f);
            ImGui::DragFloat3("Pitch / yaw / roll (deg)", cameraRotation, 0.5f);
            ImGui::SliderFloat("Field of view", &fov, 20, 100);
            if (ImGui::Button("Reset camera")) {
                cameraPosition[0] = 0; cameraPosition[1] = 1.5f; cameraPosition[2] = 9;
                cameraRotation[0] = -9; cameraRotation[1] = cameraRotation[2] = 0; fov = 55;
            }
        }
        if (ImGui::CollapsingHeader("Models and animation")) {
            ImGui::DragFloat3("Position", position, 0.02f);
            ImGui::DragFloat3("Rotation (deg)", rotation, 0.5f);
            ImGui::DragFloat3("Scale", size, 0.01f, 0.1f, 4, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::Checkbox("Second object", &secondVisible);
            if (secondVisible) ImGui::DragFloat3("Second offset", secondOffset, 0.02f);
            ImGui::Checkbox("Pause animation", &paused);
            ImGui::SliderFloat("Orbit speed", &speed, 0, 4);
            ImGui::SliderFloat("Orbit radius", &radius, 0, 2);
            ImGui::Checkbox("Procedural vertex colors", &vertexColors);
        }
        if (ImGui::CollapsingHeader("Materials and textures",ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < (secondVisible ? 2u : 1u); ++i) {
                ImGui::PushID(static_cast<int>(i));
                ImGui::Text("Material %d", static_cast<int>(i + 1));
                ImGui::Combo("Texture",&textureChoice[i],"Dragons among clouds\0Dragon medallions\0");
                ImGui::Checkbox("Animated sampling",&samplingEffect[i]);
                ImGui::ColorEdit3("Albedo tint", materials[i].albedo);
                ImGui::ColorEdit3("Specular color", materials[i].specular);
                ImGui::SliderFloat("Shininess", &materials[i].shininess, 1, 256);
                ImGui::PopID();
            }
        }
        if(ImGui::CollapsingHeader("Texture effects",ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Animate texture",&animateTexture);
            ImGui::SliderFloat("Effect speed",&effectSpeed,0,3);
            ImGui::SliderFloat("UV distortion",&effectStrength,0,1);
            ImGui::TextWrapped("Photographs of Chinese silk from The Met Museum.");
        }
        if (ImGui::CollapsingHeader("Ambient and directional light")) {
            ImGui::ColorEdit3("Ambient color", ambientColor);
            ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0, 1);
            ImGui::Checkbox("Directional enabled", &directionalEnabled);
            ImGui::DragFloat3("Ray direction", direction, 0.02f);
            ImGui::ColorEdit3("Directional color", directionalColor);
            ImGui::SliderFloat("Directional intensity", &directionalIntensity, 0, 3);
        }
        if (ImGui::CollapsingHeader("Point lights")) {
            if (ImGui::Button("Add point")) pointLights.push_back({ {0,2,2,0},{1,1,1,6} });
            for (size_t i = 0; i < pointLights.size();) {
                ImGui::PushID(static_cast<int>(i));
                bool removed = false;
                if (ImGui::TreeNode("point", "Point %d", static_cast<int>(i + 1))) {
                    auto& l = pointLights[i];
                    ImGui::DragFloat3("Position", &l.position.x, 0.05f);
                    ImGui::ColorEdit3("Color", &l.color.x);
                    ImGui::SliderFloat("Intensity", &l.color.w, 0, 50);
                    removed = ImGui::Button("Delete point");
                    ImGui::TreePop();
                }
                ImGui::PopID();
                if (removed) pointLights.erase(pointLights.begin() + i); else ++i;
            }
        }
        if (ImGui::CollapsingHeader("Spot lights")) {
            if (ImGui::Button("Add spot")) spotLights.push_back({ {0,3,3,0},{0,-1,-1,0},{1,1,1,15},{15,25,0,0} });
            for (size_t i = 0; i < spotLights.size();) {
                ImGui::PushID(static_cast<int>(i));
                bool removed = false;
                if (ImGui::TreeNode("spot", "Spot %d", static_cast<int>(i + 1))) {
                    auto& l = spotLights[i];
                    ImGui::DragFloat3("Position", &l.position.x, 0.05f);
                    ImGui::DragFloat3("Ray direction", &l.direction.x, 0.02f);
                    ImGui::ColorEdit3("Color", &l.color.x);
                    ImGui::SliderFloat("Intensity", &l.color.w, 0, 100);
                    ImGui::SliderFloat("Inner half-angle (deg)", &l.angles.x, 0, 80);
                    ImGui::SliderFloat("Outer half-angle (deg)", &l.angles.y, 0.1f, 85);
                    l.angles.y = std::max(l.angles.y, l.angles.x);
                    removed = ImGui::Button("Delete spot");
                    ImGui::TreePop();
                }
                ImGui::PopID();
                if (removed) spotLights.erase(spotLights.begin() + i); else ++i;
            }
        }
        ImGui::Checkbox("Show light markers", &showLights);
        ImGui::Checkbox("Show spot cones", &showCones);
        ImGui::SliderFloat("Cone display length", &coneLength, 0.3f, 5);
        ImGui::TextUnformatted("Cones are guides, not shadows. Angles are half-angles.");
        ImGui::PopItemWidth();
        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;
        if (!fd.command_buffer || !fd.framebuffer || !ctx.swapchain_extent.height) return;
        float aspect = static_cast<float>(ctx.swapchain_extent.width) / ctx.swapchain_extent.height;
        Mat4 projection = usePerspective ? perspective(aspect, fov) : orthographic(aspect);
        Vec3 sun = normalize({ direction[0],direction[1],direction[2] });
        SceneUniform scene{ multiply(projection,cameraView()),
            {cameraPosition[0],cameraPosition[1],cameraPosition[2],1},
            {ambientColor[0],ambientColor[1],ambientColor[2],ambientIntensity},
            {sun.x,sun.y,sun.z,0},
            {directionalColor[0],directionalColor[1],directionalColor[2],directionalEnabled ? directionalIntensity : 0},
            {static_cast<int32_t>(pointLights.size()),static_cast<int32_t>(spotLights.size()),0,0} };
        // prepare() уже дождалась fence: CPU не перезаписывает память, которую читает GPU.
        ensureStorage(pointBuffer, pointCapacity, pointLights.size(), sizeof(PointLight), 1);
        ensureStorage(spotBuffer, spotCapacity, spotLights.size(), sizeof(SpotLight), 2);
        require(writeBuffer(sceneBuffer, &scene, sizeof(scene)), "Scene upload");
        if (!pointLights.empty()) require(writeBuffer(pointBuffer, pointLights.data(), pointLights.size() * sizeof(PointLight)), "Point upload");
        if (!spotLights.empty()) require(writeBuffer(spotBuffer, spotLights.data(), spotLights.size() * sizeof(SpotLight)), "Spot upload");
        Mat4 r = objectRotation();
        Mat4 base = multiply(translate(position[0] + radius * std::cos(phase), position[1] + 0.35f * std::sin(2 * phase),
            position[2] + radius * std::sin(phase)), multiply(r, scale(size[0], size[1], size[2])));
        const Mat4 models[] = { base,multiply(translate(secondOffset[0],secondOffset[1],secondOffset[2]),base) };
        // Нормали умножаются на (M^-1)^T. Для M=T*R*S это R*S^-1, перенос не нужен.
        Mat4 normal = multiply(r, scale(1 / size[0], 1 / size[1], 1 / size[2]));
        size_t count = secondVisible ? 2 : 1;
        size_t markerCount=showLights?pointLights.size()+spotLights.size():0;
        ensureModels(count+markerCount+1);
        for (size_t i = 0; i < count; ++i) {
            auto& mat = materials[i];
            ModelUniform model{ models[i],normal,{mat.albedo[0],mat.albedo[1],mat.albedo[2],1},
                {mat.specular[0],mat.specular[1],mat.specular[2],1},
                {mat.shininess,vertexColors ? 1.f : 0.f,0,0} };
            require(writeBuffer(modelBuffer, &model, sizeof(model),modelStride*i), "Model upload");
        }
        markers.clear(); coneVertices.clear();
        auto marker = [](Vec4 p, Vec4 color) {
            markers.push_back({ multiply(translate(p.x,p.y,p.z),scale(0.125f,0.2f,0.25f)),
                {color.x,color.y,color.z,1} });
            };
        if (showLights) {
            for (auto& l : pointLights) marker(l.position, l.color);
            for (auto& l : spotLights) marker(l.position, l.color);
        }
        for(size_t i=0;i<markers.size();++i) {
            ModelUniform model{markers[i].model,identity(),markers[i].color,{0,0,0,0},{1,0,0,0}};
            require(writeBuffer(modelBuffer,&model,sizeof(model),modelStride*(count+i)),"Marker upload");
        }
        ModelUniform lines{identity(),identity(),{1,1,1,1},{0,0,0,0},{1,0,0,0}};
        require(writeBuffer(modelBuffer,&lines,sizeof(lines),modelStride*(count+markers.size())),"Line model upload");
        if (showCones) for (auto& l : spotLights) makeCone(l);
        if (!coneVertices.empty()) {
            size_t bytes = coneVertices.size() * sizeof(Vertex);
            if (bytes > lineCapacity) {
                destroyBuffer(lineBuffer); lineCapacity = bytes;
            }
            uploadGpu(lineBuffer,coneVertices.data(),bytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        }
        require(vkResetCommandBuffer(fd.command_buffer, 0) == VK_SUCCESS, "Command reset");
        const VkCommandBufferBeginInfo begin{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        require(vkBeginCommandBuffer(fd.command_buffer, &begin) == VK_SUCCESS, "Command begin");
        const VkClearValue clear[] = { {.color = {{0.025f,0.035f,0.055f,1}}}, {.depthStencil = {1,0}} };
        const VkRenderPassBeginInfo pass{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = ctx.render_pass,.framebuffer = fd.framebuffer,.renderArea = {{0,0},ctx.swapchain_extent},
            .clearValueCount = 2,.pClearValues = clear };
        vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport vp{ 0,static_cast<float>(ctx.swapchain_extent.height),static_cast<float>(ctx.swapchain_extent.width),
            -static_cast<float>(ctx.swapchain_extent.height),0,1 };
        const VkRect2D scissor{ {0,0},ctx.swapchain_extent };
        vkCmdSetViewport(fd.command_buffer, 0, 1, &vp); vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer.handle, &offset);
        // Только параметры эффектов и флаг unlit. Все матрицы и материалы находятся в UBO.
        struct Push { Vec4 effect; };
        static_assert(sizeof(Push) == 16);
        Push push{ {0,effectTime,effectStrength,0} };
        vkCmdBindIndexBuffer(fd.command_buffer,indexBuffer.handle,0,VK_INDEX_TYPE_UINT16);
        vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        for (size_t i = 0; i < count; ++i) {
            const VkDescriptorSet sets[] = { sceneSet,materialSets[textureChoice[i]] };
            const uint32_t dynamicOffset=static_cast<uint32_t>(modelStride*i);
            push.effect.x=samplingEffect[i]?1.f:0.f;
            vkCmdPushConstants(fd.command_buffer,pipelineLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(push),&push);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 1, &dynamicOffset);
            vkCmdDrawIndexed(fd.command_buffer,static_cast<uint32_t>(indices.size()),1,0,0,0);
        }
        // Маркеры используют общий dynamic UBO, флаг effect.w отключает освещение.
        push.effect.w=1;
        vkCmdPushConstants(fd.command_buffer,pipelineLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(push),&push);
        const VkDescriptorSet markerSets[]={sceneSet,materialSets[0]};
        for(size_t i=0;i<markers.size();++i) {
            const uint32_t dynamicOffset=static_cast<uint32_t>(modelStride*(count+i));
            vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,0,2,markerSets,1,&dynamicOffset);
            vkCmdDrawIndexed(fd.command_buffer,static_cast<uint32_t>(indices.size()),1,0,0,0);
        }
        if (!coneVertices.empty()) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &lineBuffer.handle, &offset);
            const uint32_t dynamicOffset=static_cast<uint32_t>(modelStride*(count+markers.size()));
            vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,0,2,markerSets,1,&dynamicOffset);
            vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(fd.command_buffer, static_cast<uint32_t>(coneVertices.size()), 1, 0, 0);
        }
        vkCmdEndRenderPass(fd.command_buffer);
        require(vkEndCommandBuffer(fd.command_buffer) == VK_SUCCESS, "Command end");
    }
} // Конец пространства имён.
