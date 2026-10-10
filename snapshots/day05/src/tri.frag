#version 460
// Interpolated across the triangle, because a varying is interpolated unless you say
// otherwise.
//
// THE ALPHA IS A SPECIALIZATION CONSTANT, which means a scene that blends at 0.45 is a
// different compiled shader from one that blends at 0.65 -- and therefore a different pipeline.
// That is the day's argument in one line: almost everything is baked in.
layout(constant_id = 0) const float ALPHA = 1.0;

layout(location = 0) in  vec3 vColour;
layout(location = 0) out vec4 oColour;

void main() {
    oColour = vec4(vColour, ALPHA);
}
