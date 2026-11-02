#version 450
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=2) in vec3 vertexColor;
layout(location=0) out vec4 outColor;
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
struct PointLight { vec4 position; vec4 color; };
struct SpotLight { vec4 position; vec4 direction; vec4 color; vec4 angles; };
// Массивы без фиксированного размера читаются из storage-буферов.
layout(std430,set=0,binding=1) readonly buffer Points { PointLight lights[]; } points;
layout(std430,set=0,binding=2) readonly buffer Spots { SpotLight lights[]; } spots;
layout(push_constant) uniform Draw { mat4 model; vec4 color; } draw;

vec3 safeNormalize(vec3 v) {
    return v*inversesqrt(max(dot(v,v),1e-10));
}
// L направлен от поверхности к свету, V — к камере. H — половинный вектор.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diffuse=max(dot(N,L),0.0);
    vec3 H=safeNormalize(L+V);
    // На стороне, отвёрнутой от источника, блик не рисуем.
    float specular=diffuse>0.0?pow(max(dot(N,H),0.0),object.parameters.x):0.0;
    return radiance*(albedo*diffuse+object.specular.rgb*specular);
}
void main() {
    // Кубики света и линии конусов остаются своего цвета при любом освещении.
    if (draw.color.w>0.5) { outColor=vec4(draw.color.rgb,1.0); return; }
    vec3 N=safeNormalize(worldNormal);
    vec3 V=safeNormalize(scene.cameraPosition.xyz-worldPosition);
    vec3 albedo=object.albedo.rgb;
    if (object.parameters.y>0.5) albedo*=vertexColor;
    vec3 result=albedo*scene.ambient.rgb*scene.ambient.w;
    result+=blinnPhong(N,V,-scene.direction.xyz,
        scene.directionalColor.rgb*scene.directionalColor.w,albedo);
    for (int i=0;i<scene.counts.x;++i) {
        PointLight light=points.lights[i];
        vec3 toLight=light.position.xyz-worldPosition;
        // I/r^2. Вблизи самой точки ограничиваем знаменатель, чтобы не делить на ноль.
        float distanceSquared=max(dot(toLight,toLight),0.01);
        vec3 radiance=light.color.rgb*light.color.w/distanceSquared;
        result+=blinnPhong(N,V,safeNormalize(toLight),radiance,albedo);
    }
    for (int i=0;i<scene.counts.y;++i) {
        SpotLight light=spots.lights[i];
        vec3 toLight=light.position.xyz-worldPosition;
        vec3 L=safeNormalize(toLight);
        vec3 axis=safeNormalize(light.direction.xyz);
        if (dot(axis,axis)<0.5) axis=vec3(0,-1,0);
        float cosine=dot(-L,axis);
        float inner=cos(radians(light.angles.x));
        float outer=cos(radians(light.angles.y));
        // Между внешним и внутренним конусами свет плавно усиливается от 0 до 1.
        // При равных углах используем чёткую границу без деления на ноль.
        float edge=inner-outer>1e-6?smoothstep(outer,inner,cosine):step(outer,cosine);
        float distanceSquared=max(dot(toLight,toLight),0.01);
        vec3 radiance=light.color.rgb*light.color.w*edge/distanceSquared;
        result+=blinnPhong(N,V,L,radiance,albedo);
    }
    outColor=vec4(result,1.0);
}
