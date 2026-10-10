#version 460
// gl_PointCoord IS THE ONE THING A POINT GIVES YOU THAT A TRIANGLE DOES NOT: where in the
// square this fragment is, 0 to 1 in both directions. Without it every mote is a hard square.
layout(location = 0) in vec4 vColour;
layout(location = 0) out vec4 oColour;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    // NO discard, AND NOT BECAUSE IT WOULD BE WRONG. Adding zero and not adding are the same
    // thing under an additive blend, so clamping the falloff at the corners does what a discard
    // would do -- without OpDemoteToHelperInvocation, which is a SPIR-V capability the device
    // has to have asked for, and without the helper lanes a discard leaves running.
    float a = max(1.0 - r2, 0.0);
    a *= a;                                 // squared, so the edge fades rather than steps
    // THE ALPHA IS FOLDED INTO THE COLOUR, because the blend is ONE, ONE. Additive blending
    // wants a number to ADD, and premultiplying is how you say "add this much of it".
    oColour = vec4(vColour.rgb * a * vColour.w, 1.0);
}
