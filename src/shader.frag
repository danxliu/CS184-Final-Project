#version 330
in vec3 vNormal;
out vec4 color;
void main() {
    vec3 n = normalize(vNormal);
    float diffuse = max(dot(n, normalize(vec3(1, 1, 1))), 0.0);
    color = vec4(vec3(0.3 + 0.7 * diffuse), 1.0);
}
