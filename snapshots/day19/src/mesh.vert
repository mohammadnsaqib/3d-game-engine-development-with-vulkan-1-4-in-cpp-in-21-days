#version 460
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 eye;
    vec4 sun;
} cam;

// THE PUSH BLOCK CARRIES A SURFACE NOW. Two floats -- roughness and metalness -- and they are
// the whole of what a modern material is. Ninety-six bytes, still inside the guaranteed 128.
layout(push_constant) uniform Push {
    mat4  model;
    vec4  tint;
    int   material;
    int   mode;
    float roughness;
    float metalness;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 vColour;
layout(location = 1) out vec2 vUV;
layout(location = 2) flat out int vMaterial;
layout(location = 3) out vec3 vNormal;
layout(location = 4) out vec3 vWorld;

void main() {
    vec4 world = pc.model * vec4(inPosition, 1.0);
    gl_Position = cam.viewProj * world;
    vColour = inColour * pc.tint.rgb;
    vUV = inUV;
    vMaterial = pc.material;
    vNormal = mat3(pc.model) * inNormal;
    vWorld = world.xyz;
}
