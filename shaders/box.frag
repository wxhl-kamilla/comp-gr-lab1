#version 450
layout(location = 0) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;
// Запись цвета фрагмента с полной непрозрачностью.
void main() {
    outColor = vec4(vertexColor, 1.0);
}

