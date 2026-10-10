#version 460
// TWO BINDINGS, AND THEY ADVANCE AT DIFFERENT RATES.
//
// Binding 0 is the star's mesh -- 122 vertices, the same 122 for every copy. Binding 1 is the
// PER-INSTANCE data, and the fetcher moves to the next element of it once per INSTANCE rather
// than once per vertex. That one enum, VK_VERTEX_INPUT_RATE_INSTANCE, is the whole mechanism.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 eye;
    vec4 sun;
} cam;

layout(push_constant) uniform Push {
    float scale;        // the scene's size multiplier
    float glow;
    float pad0, pad1;
} pc;

// ---- binding 0: per vertex ----
// LOCATIONS 1 AND 2 ARE MISSING ON PURPOSE. The Vertex struct has a colour and a UV in it, and
// a star wants neither -- so this shader does not declare them and the pipeline below does not
// describe them. Declaring an attribute the shader never reads is a validation WARNING, which
// is the layer saying you have described bytes nobody asked for.
layout(location = 0) in vec3 inPosition;
layout(location = 3) in vec3 inNormal;
// ---- binding 1: per instance ----
layout(location = 4) in vec4 iPosScale;
layout(location = 5) in vec4 iColourGlow;

layout(location = 0) out vec3 vColour;
layout(location = 1) out vec3 vNormal;

void main() {
    // NO MODEL MATRIX. A translation and a uniform scale is four floats, and sending sixteen
    // instead would be four times the bandwidth for the same picture. An instance that needs a
    // full rotation sends a matrix; most do not.
    vec3 world = inPosition * (iPosScale.w * pc.scale) + iPosScale.xyz;
    gl_Position = cam.viewProj * vec4(world, 1.0);
    vNormal = inNormal;
    vColour = iColourGlow.rgb * (iColourGlow.w * pc.glow);
}
