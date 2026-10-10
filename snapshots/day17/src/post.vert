#version 460
// A FULL-SCREEN TRIANGLE, AND NOT A QUAD. One triangle whose corners are at (-1,-1), (3,-1) and
// (-1,3) covers the whole screen with no diagonal seam down the middle -- which matters because
// a quad's two triangles meet along that diagonal and the hardware rasterizes the shared edge
// twice, running the fragment shader on the same pixels in two different waves.
//
// NO VERTEX BUFFER. Three vertices out of gl_VertexIndex, which is where Day 1 started and the
// one place in a finished engine where it is still the right answer.
layout(location = 0) out vec2 vUV;

void main() {
    vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
