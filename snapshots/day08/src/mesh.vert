#version 460
// THE PUSH BLOCK HAS TWO MEMBERS NOW, and it is still one block: 64 bytes of matrix and 8 of
// texture scale, well inside the 128 that every Vulkan implementation guarantees.
layout(push_constant) uniform Push {
    mat4 mvp;
    vec2 uvScale;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vColour;
layout(location = 1) out vec2 vUV;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vColour = inColour;
    // SCALING HERE RATHER THAN IN THE MESH is what lets one plane serve four address modes.
    // Coordinates outside 0..1 are not an error; what happens to them is the sampler's job.
    vUV = inUV * pc.uvScale;
}
