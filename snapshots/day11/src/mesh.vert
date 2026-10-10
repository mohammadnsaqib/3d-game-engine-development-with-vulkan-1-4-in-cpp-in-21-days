#version 460
// THE PUSH BLOCK CARRIES TWO MATRICES NOW, and that is the whole of today's argument about
// where arithmetic goes. A normal must be rotated by the MODEL matrix and nothing else: put it
// through the projection and it stops being a direction and becomes a point, which is a bug
// that looks like bad lighting rather than like a bug.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 eye;
} cam;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normalMatrix;
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
    // MODE 4 IS THE DELIBERATE MISTAKE: the normal through the full transform, which is what
    // happens when somebody reuses the matrix that was already to hand.
    vNormal = pc.mode == 4 ? mat3(cam.viewProj * pc.model) * inNormal
                           : mat3(pc.normalMatrix) * inNormal;
    vWorld = world.xyz;
}
