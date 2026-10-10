#version 460
// THE SAME TRIANGLE, WITH OPENGL'S Y AXIS, which is the mistake everybody makes once.
//
// These are exactly the numbers that draw an upright triangle in OpenGL and in Direct3D. In
// Vulkan they draw it upside down, because +Y goes down the screen here. The engine renders
// this shader as its own scene so that the chapter can print both pictures rather than assert
// the difference.
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
