#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 position;
in vec3 normal;
out vec3 vNormal;
void main() {
    vNormal = mat3(model) * normal;
    gl_Position = mvp * vec4(position, 1.0);
}
