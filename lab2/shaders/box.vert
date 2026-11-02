#version 450
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec3 inColor;

// Набор 0 общий для всей сцены, набор 1 меняется при выборе модели.
layout(std140,set=0,binding=0) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 direction;
    vec4 directionalColor;
    ivec4 counts;
} scene;
layout(std140,set=1,binding=0) uniform Model {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
    vec4 parameters;
} object;
layout(push_constant) uniform Draw {
    mat4 model;
    vec4 color;
} draw;
layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(location=2) out vec3 vertexColor;

void main() {
    // У маркеров color.w=1: их матрица передаётся командой push constants.
    bool unlit=draw.color.w>0.5;
    vec4 p=(unlit?draw.model:object.model)*vec4(inPosition,1.0);
    worldPosition=p.xyz;
    worldNormal=unlit?inNormal:mat3(object.normalMatrix)*inNormal;
    vertexColor=inColor;
    gl_Position=scene.viewProjection*p;
}
