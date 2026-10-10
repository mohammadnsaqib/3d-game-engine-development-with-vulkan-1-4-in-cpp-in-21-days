#version 460
// THE SAME BYTES THE COMPUTE SHADER WROTE, READ AS VERTEX ATTRIBUTES. One buffer, created with
// both STORAGE_BUFFER and VERTEX_BUFFER in its usage flags, and nothing copies between them.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 eye;
    vec4 sun;
} cam;

layout(push_constant) uniform Push {
    float size;         // the scene's size multiplier
    float glow;         // how bright, and therefore whether it blooms
    float pad0, pad1;
} pc;

layout(location = 0) in vec4 aPosSize;
layout(location = 1) in vec4 aColourLife;

layout(location = 0) out vec4 vColour;

void main() {
    vec4 world = vec4(aPosSize.xyz, 1.0);
    gl_Position = cam.viewProj * world;
    // A POINT IS A SQUARE OF PIXELS AND ITS SIZE IS IN PIXELS, not in world units -- which
    // means a mote further away has to be made smaller by hand. The perspective divide does
    // this for triangles and does not do it for points.
    float d = distance(cam.eye.xyz, world.xyz);
    gl_PointSize = clamp(pc.size * aPosSize.w * 46.0 / max(d, 0.75), 1.0, 63.0);
    vColour = vec4(aColourLife.rgb * pc.glow, aColourLife.w);
}
