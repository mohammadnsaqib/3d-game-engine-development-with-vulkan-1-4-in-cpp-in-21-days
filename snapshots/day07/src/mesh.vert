#version 460
// A PUSH CONSTANT IS THE SMALLEST WAY TO GIVE A SHADER A NUMBER. No buffer, no descriptor, no
// allocation: the values ride along inside the command buffer itself, and the guaranteed size
// is 128 bytes on every Vulkan implementation -- exactly two mat4s, which is not an accident.
layout(push_constant) uniform Push {
    mat4 mvp;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColour;

layout(location = 0) out vec3 vColour;

void main() {
    // ONE MULTIPLICATION, AND IT CONTAINS EVERYTHING: where the object is, where the camera
    // is, what lens it has, and the y flip that has caught this book out three times.
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vColour = inColour;
}
