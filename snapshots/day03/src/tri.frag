#version 460
// Interpolated across the triangle, because a varying is interpolated unless you say
// otherwise. Day 5 is about what "otherwise" means.
layout(location = 0) in  vec3 vColour;
layout(location = 0) out vec4 oColour;

void main() {
    oColour = vec4(vColour, 1.0);
}
