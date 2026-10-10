#version 460
// THE TRIANGLE, WITH VULKAN'S Y AXIS. No vertex buffer: three corners from gl_VertexIndex,
// because a buffer is Day 4's subject and a picture is today's.
//
// NOTE THE SIGNS. Vulkan's clip space has +Y pointing DOWN the screen, where OpenGL's points up
// and Direct3D's points up. So the apex of an upright triangle has a NEGATIVE y here. Nothing
// warns you about this; the picture simply comes out upside down, and naive.vert is that
// picture.
layout(location = 0) out vec3 vColour;

void main() {
    const vec2 P[3] = vec2[3](vec2( 0.00, -0.72),
                              vec2(-0.82,  0.58),
                              vec2( 0.82,  0.58));
    const vec3 C[3] = vec3[3](vec3(0.98, 0.72, 0.32),
                              vec3(0.36, 0.52, 0.92),
                              vec3(0.86, 0.36, 0.30));
    gl_Position = vec4(P[gl_VertexIndex], 0.0, 1.0);
    vColour = C[gl_VertexIndex];

    // AND A SIZE FOR THE POINT THIS MIGHT BECOME. A pipeline whose polygonMode is POINT
    // rasterises each corner as a square of gl_PointSize pixels, and a shader that does not
    // write one draws NOTHING -- no error from the driver, an empty picture, and the
    // validation layer as the only thing in the world that will say so. It is ignored when
    // the polygon mode is FILL or LINE, so writing it always costs nothing.
    gl_PointSize = 96.0;
}
