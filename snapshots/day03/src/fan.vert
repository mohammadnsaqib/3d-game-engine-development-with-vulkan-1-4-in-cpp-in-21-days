#version 460
// A FAN OF WEDGES, AND ONE DRAW CALL EACH.
//
// Each wedge is three vertices: the middle, and two points on a circle. The engine issues one
// vkCmdDraw per wedge with firstVertex set to 3 * segment, so gl_VertexIndex arrives already
// carrying which wedge this is -- which is how today's chapter gets a scene with two hundred
// and fifty-six draw calls in it without a buffer or a descriptor anywhere.
//
// SPECIALIZATION CONSTANTS, which are the day's second subject. These three look like
// compile-time constants and are not: their values are supplied when the PIPELINE is created,
// and the driver compiles the SPIR-V again for each set of them. A constant is therefore free
// at run time -- the loop bound below is a literal by the time the shader runs -- and costs a
// pipeline. That is the trade, and Vulkan makes you see it.
layout(constant_id = 0) const int   SEGMENTS = 64;
layout(constant_id = 1) const int   MODE     = 0;     // 0 disc, 1 star, 2 spiral
layout(constant_id = 2) const float ASPECT   = 1.7778;

layout(location = 0) out vec3 vColour;

const float TAU = 6.28318530718;

float radius_at(int seg) {
    if (MODE == 1) return (seg % 2 == 0) ? 0.86 : 0.42;                  // a star
    if (MODE == 2) return 0.20 + 0.66 * float(seg) / float(SEGMENTS);    // a spiral
    return 0.86;                                                        // a disc
}

// A hue wheel, so the fan is a gradient rather than a flat colour and a reader can see where
// one wedge ends and the next begins. Cosines a third of a turn apart make red, green and blue
// peak at three evenly spaced angles.
vec3 hue(float t) {
    return 0.55 + 0.45 * cos(TAU * (t + vec3(0.00, 0.33, 0.67)));
}

void main() {
    int seg    = gl_VertexIndex / 3;
    int corner = gl_VertexIndex % 3;

    float a0 = TAU * float(seg)     / float(SEGMENTS);
    float a1 = TAU * float(seg + 1) / float(SEGMENTS);
    float r0 = radius_at(seg);
    float r1 = radius_at(seg + 1);

    vec2 p = vec2(0.0);
    if (corner == 1) p = vec2(r0 * cos(a0), r0 * sin(a0));
    if (corner == 2) p = vec2(r1 * cos(a1), r1 * sin(a1));

    // THE IMAGE IS NOT SQUARE AND CLIP SPACE IS. Dividing x by the aspect ratio keeps a circle
    // a circle; without it a 1920 by 1080 image draws an ellipse. Day 6 does this properly, in
    // a projection matrix, for everything at once.
    p.x /= ASPECT;

    gl_Position = vec4(p, 0.0, 1.0);
    gl_PointSize = 12.0;
    vColour = (corner == 0) ? vec3(0.10, 0.11, 0.16) : hue(float(seg) / float(SEGMENTS));
}
