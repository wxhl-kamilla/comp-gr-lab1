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
        struct Vertex { Vec3 position; Vec3 normal; Vec3 color; };
        std::vector<Vertex> vertices;

        // У каждой грани свои вершины: в одном углу коробки встречаются три разные нормали.
        void makeBox() {
            std::array<Vec3, 8> corners{};
            for (int i = 0; i < 8; ++i)
                corners[i] = { (i & 1) ? 0.8f : -0.8f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.4f : -0.4f };
            const int indices[] = { 4,5,7,4,7,6, 1,0,2,1,2,3, 0,4,6,0,6,2,
                                   5,1,3,5,3,7, 2,6,7,2,7,3, 0,1,5,0,5,4 };
            for (int i = 0; i < 36; i += 3) {
                Vec3 a = corners[indices[i]], b = corners[indices[i + 1]], c = corners[indices[i + 2]];
                // Векторное произведение двух рёбер даёт нормаль треугольника наружу.
                Vec3 n = normalize(cross(b - a, c - a));
                for (int j = 0; j < 3; ++j) {
                    Vec3 v = corners[indices[i + j]];
                    vertices.push_back({ v,n,{0.25f + 0.75f * (v.x + 0.8f) / 1.6f,
                        0.25f + 0.75f * (v.y + 0.5f),0.25f + 0.75f * (v.z + 0.4f) / 0.8f} });
                }
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
        std::array<Buffer, 2> modelBuffers{};
        size_t pointCapacity = 0, spotCapacity = 0, lineCapacity = 0;
        VkDescriptorSetLayout sceneLayout = VK_NULL_HANDLE, modelLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet sceneSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> modelSets{};
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
            float albedo[3] = { 0.55f,0.7f,0.9f };
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
        void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
            const Buffer& buffer, VkDeviceSize bytes) {
            VkDescriptorBufferInfo info{ buffer.handle,0,bytes };
            VkWriteDescriptorSet write{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,.dstBinding = binding,.descriptorCount = 1,
                .descriptorType = type,.pBufferInfo = &info };
            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
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
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))}
            };
            const VkPipelineVertexInputStateCreateInfo input{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
                .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes };
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
            require(createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer), "Vertex buffer");
            require(writeBuffer(vertexBuffer, vertices.data(), vertices.size() * sizeof(Vertex)), "Vertex upload");
            require(createBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneBuffer), "Scene buffer");
            for (auto& b : modelBuffers)
                require(createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, b), "Model buffer");
            // set=0: общая сцена, массив точечных источников и массив прожекторов.
            const VkDescriptorSetLayoutBinding sceneBindings[] = {
                {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr} };
            VkDescriptorSetLayoutCreateInfo layout{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,.pBindings = sceneBindings };
            require(vkCreateDescriptorSetLayout(ctx.device, &layout, nullptr, &sceneLayout) == VK_SUCCESS, "Scene layout");
            // set=1: матрица и материал конкретной модели; у двух коробок отдельные наборы.
            const VkDescriptorSetLayoutBinding modelBinding{ 0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,nullptr };
            layout.bindingCount = 1; layout.pBindings = &modelBinding;
            require(vkCreateDescriptorSetLayout(ctx.device, &layout, nullptr, &modelLayout) == VK_SUCCESS, "Model layout");
            const VkDescriptorSetLayout layouts[] = { sceneLayout,modelLayout };
            const VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,0,80 };
            const VkPipelineLayoutCreateInfo pipelineInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,.pSetLayouts = layouts,.pushConstantRangeCount = 1,.pPushConstantRanges = &push };
            require(vkCreatePipelineLayout(ctx.device, &pipelineInfo, nullptr, &pipelineLayout) == VK_SUCCESS, "Pipeline layout");
            const VkDescriptorPoolSize poolSizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,3},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2} };
            const VkDescriptorPoolCreateInfo poolInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 3,.poolSizeCount = 2,.pPoolSizes = poolSizes };
            require(vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) == VK_SUCCESS, "Descriptor pool");
            const VkDescriptorSetLayout allocatedLayouts[] = { sceneLayout,modelLayout,modelLayout };
            const VkDescriptorSetAllocateInfo allocation{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptorPool,.descriptorSetCount = 3,.pSetLayouts = allocatedLayouts };
            VkDescriptorSet allocated[3]{};
            require(vkAllocateDescriptorSets(ctx.device, &allocation, allocated) == VK_SUCCESS, "Descriptor sets");
            sceneSet = allocated[0]; modelSets = { allocated[1],allocated[2] };
            bindBuffer(sceneSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sceneBuffer, sizeof(SceneUniform));
            for (size_t i = 0; i < 2; ++i)
                bindBuffer(modelSets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, modelBuffers[i], sizeof(ModelUniform));
            // Даже при нулевом числе источников дескриптор должен указывать на настоящий буфер.
            ensureStorage(pointBuffer, pointCapacity, 1, sizeof(PointLight), 1);
            ensureStorage(spotBuffer, spotCapacity, 1, sizeof(SpotLight), 2);
            require(createPipeline(false, pipeline), "Triangle pipeline");
            require(createPipeline(true, linePipeline), "Line pipeline");
            pointLights.push_back({ {-0.5f,2.2f,2,0},{0.25f,0.55f,1,9} });
            spotLights.push_back({ {3,3,3,0},{-0.6f,-0.6f,-1,0},{1,0.4f,0.15f,20},{15,25,0,0} });
            materials[1].albedo[0] = 0.85f; materials[1].albedo[1] = 0.45f; materials[1].albedo[2] = 0.25f;
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
        for (auto& b : modelBuffers) destroyBuffer(b);
        for (Buffer* b : { &vertexBuffer,&sceneBuffer,&pointBuffer,&spotBuffer,&lineBuffer }) destroyBuffer(*b);
        pipeline = linePipeline = VK_NULL_HANDLE; pipelineLayout = VK_NULL_HANDLE;
        descriptorPool = VK_NULL_HANDLE; sceneLayout = modelLayout = VK_NULL_HANDLE;
        pointCapacity = spotCapacity = lineCapacity = 0;
        vertices.clear(); pointLights.clear(); spotLights.clear(); previousTime = -1;
    }

    void update(double time) {
        if (previousTime < 0) previousTime = time;
        float delta = static_cast<float>(std::clamp(time - previousTime, 0.0, 0.1));
        previousTime = time;
        if (!paused) phase += speed * delta;
        ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(470, 690), ImGuiCond_FirstUseEver);
        ImGui::Begin("Lab 2: camera and lighting - variant 3");
        ImGui::PushItemWidth(225);
        // ImGui сразу меняет эти переменные. Матрицы и буферы обновятся в render этого кадра.
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            for (size_t i = 0; i < (secondVisible ? 2u : 1u); ++i) {
                ImGui::PushID(static_cast<int>(i));
                ImGui::Text("Material %d", static_cast<int>(i + 1));
                ImGui::ColorEdit3("Albedo", materials[i].albedo);
                ImGui::ColorEdit3("Specular color", materials[i].specular);
                ImGui::SliderFloat("Shininess", &materials[i].shininess, 1, 256);
                ImGui::PopID();
            }
        }
        if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient color", ambientColor);
            ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0, 1);
            ImGui::Checkbox("Directional enabled", &directionalEnabled);
            ImGui::DragFloat3("Ray direction", direction, 0.02f);
            ImGui::ColorEdit3("Directional color", directionalColor);
            ImGui::SliderFloat("Directional intensity", &directionalIntensity, 0, 3);
        }
        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        for (size_t i = 0; i < count; ++i) {
            auto& mat = materials[i];
            ModelUniform model{ models[i],normal,{mat.albedo[0],mat.albedo[1],mat.albedo[2],1},
                {mat.specular[0],mat.specular[1],mat.specular[2],1},
                {mat.shininess,vertexColors ? 1.f : 0.f,0,0} };
            require(writeBuffer(modelBuffers[i], &model, sizeof(model)), "Model upload");
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
        if (showCones) for (auto& l : spotLights) makeCone(l);
        if (!coneVertices.empty()) {
            size_t bytes = coneVertices.size() * sizeof(Vertex);
            if (bytes > lineCapacity) {
                Buffer replacement{};
                require(createBuffer(bytes * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, replacement), "Line buffer");
                destroyBuffer(lineBuffer); lineBuffer = replacement; lineCapacity = bytes * 2;
            }
            require(writeBuffer(lineBuffer, coneVertices.data(), bytes), "Line upload");
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
        // Push constants хранят режим рисования и матрицу маркера. 80 байт укладываются в минимум Vulkan 128.
        struct Push { Mat4 model; Vec4 color; };
        static_assert(sizeof(Push) == 80);
        Push push{ identity(),{0,0,0,0} };
        vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        for (size_t i = 0; i < count; ++i) {
            const VkDescriptorSet sets[] = { sceneSet,modelSets[i] };
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);
            vkCmdDraw(fd.command_buffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        }
        // Маркеры используют тот же шейдер, но флаг color.w=1 отключает расчёт освещения.
        for (auto& m : markers) {
            push = { m.model,m.color };
            vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(fd.command_buffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        }
        if (!coneVertices.empty()) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &lineBuffer.handle, &offset);
            push = { identity(),{1,1,1,1} };
            vkCmdPushConstants(fd.command_buffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDraw(fd.command_buffer, static_cast<uint32_t>(coneVertices.size()), 1, 0, 0);
        }
        vkCmdEndRenderPass(fd.command_buffer);
        require(vkEndCommandBuffer(fd.command_buffer) == VK_SUCCESS, "Command end");
    }
} // Конец пространства имён.
