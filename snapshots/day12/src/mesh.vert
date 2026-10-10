#version 460
// THE NORMAL MATRIX IS GONE, and with it sixty-four wasted bytes. Yesterday this engine sent the
// model matrix twice under two names, because a normal needs a 3x3 and the model matrix's own
// 3x3 is the right one for a rotation and a uniform scale. mat3(pc.model) says that directly,
// and the push block drops from 152 bytes to 88 -- back inside the 128 every implementation
// guarantees, which yesterday's version was not.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 eye;
} cam;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 tint;
    int  material;
    int  mode;
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
    // THE WORLD POSITION IS WHAT MAKES A LAMP POSSIBLE. A directional light needs only a
    // normal; a lamp needs to know WHERE the fragment is, because the direction to the light
    // and the distance to it are both different at every pixel.
    vWorld = world.xyz;
}
