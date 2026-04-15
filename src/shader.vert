#version 330 core
in vec3 position;
in vec3 normal;

uniform mat4 mvp;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
}
