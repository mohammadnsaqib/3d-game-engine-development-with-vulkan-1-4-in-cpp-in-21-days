#version 460
// SET 0 IS THE CAMERA, AND IT IS BOUND ONCE FOR THE WHOLE FRAME. Nothing in it changes between
// objects, which is exactly the rule that decides which set a thing belongs in: SET NUMBER IS
// BINDING FREQUENCY, and a descriptor that changes less often belongs in a lower-numbered set.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 eye;
} cam;

// AND THE PUSH CONSTANT IS NOW THE PER-OBJECT DATA. A model matrix, a material index, and a
// tint: 84 bytes, still inside the 128 every implementation guarantees.
layout(push_constant) uniform Push {
    mat4 model;
    vec4 tint;
    int  material;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vColour;
layout(location = 1) out vec2 vUV;
layout(location = 2) flat out int vMaterial;

void main() {
    gl_Position = cam.viewProj * pc.model * vec4(inPosition, 1.0);
    vColour = inColour * pc.tint.rgb;
    vUV = inUV;
    // flat, because an integer cannot be interpolated and the compiler will not guess.
    vMaterial = pc.material;
}
