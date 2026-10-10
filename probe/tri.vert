#version 460
// No vertex buffer. Three corners from the vertex index, which is the smallest thing that can
// draw and the right way to start: a buffer is Day 4's subject, not Day 1's.
layout(location = 0) out vec3 vColour;

void main() {
    const vec2 P[3] = vec2[3](vec2( 0.00,  0.72),
                              vec2(-0.82, -0.58),
                              vec2( 0.82, -0.58));
    const vec3 C[3] = vec3[3](vec3(0.98, 0.72, 0.32),
                              vec3(0.36, 0.52, 0.92),
                              vec3(0.86, 0.36, 0.30));
    gl_Position = vec4(P[gl_VertexIndex], 0.0, 1.0);
    vColour = C[gl_VertexIndex];
}
