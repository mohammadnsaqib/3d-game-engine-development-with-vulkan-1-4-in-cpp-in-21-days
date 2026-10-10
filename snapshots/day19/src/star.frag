#version 460
// A STAR IS NOT LIT. It is the thing doing the lighting, so it has a colour and a rim rather
// than a BRDF -- which also keeps the instanced pass off the descriptor set the Meridian uses.
layout(location = 0) in vec3 vColour;
layout(location = 1) in vec3 vNormal;
layout(location = 0) out vec4 oColour;

void main() {
    // A CHEAP RIM: brighter where the surface turns away, so a sphere reads as a sphere
    // instead of a flat disc, at the cost of one dot product.
    float rim = 0.55 + 0.45 * abs(normalize(vNormal).z);
    oColour = vec4(vColour * rim, 1.0);
}
