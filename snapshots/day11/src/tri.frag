#version 460
// THE FIRST SHADING IN THIS BOOK. Everything until now has been paint: a colour fetched from a
// texture and multiplied by a tint, identical whichever way the surface faced.
//
// LAMBERT IS ONE DOT PRODUCT. How much light a matte surface receives is the cosine of the
// angle between its normal and the direction to the light, and a dot product of two unit
// vectors IS that cosine. Everything else in this shader is bookkeeping around that one line.
layout(constant_id = 0) const float ALPHA = 1.0;
layout(constant_id = 1) const int MATERIAL_COUNT = 8;

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 eye;
} cam;

layout(set = 1, binding = 0) uniform sampler2D uMaterials[8];

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normalMatrix;
    vec4 tint;
    int  material;
    int  mode;
} pc;

layout(location = 0) in  vec3 vColour;
layout(location = 1) in  vec2 vUV;
layout(location = 2) flat in int vMaterial;
layout(location = 3) in  vec3 vNormal;
layout(location = 4) in  vec3 vWorld;
layout(location = 0) out vec4 oColour;

// THE LIGHT IS A DIRECTION, NOT A PLACE. A directional light is infinitely far away, so every
// point in the scene receives it from the same angle -- which is the sun, and which is why it
// needs no falloff and no position. Day 12 puts lamps in the room.
layout(constant_id = 2) const float LIGHT_X = -0.45;
layout(constant_id = 3) const float LIGHT_Y = 0.82;
layout(constant_id = 4) const float LIGHT_Z = 0.36;
layout(constant_id = 5) const float AMBIENT = 0.16;

void main() {
    vec3 albedo = texture(uMaterials[vMaterial % MATERIAL_COUNT], vUV).rgb * vColour;

    // RENORMALISE. The normal was unit length at the vertex and is not unit length after
    // interpolation across a triangle -- linear interpolation of two unit vectors gives
    // something shorter than either, and the error is worst exactly where the curvature is.
    vec3 n = normalize(vNormal);
    vec3 l = normalize(vec3(LIGHT_X, LIGHT_Y, LIGHT_Z));

    if (pc.mode == 1) { oColour = vec4(albedo, ALPHA); return; }          // unlit
    if (pc.mode == 2) { oColour = vec4(n * 0.5 + 0.5, ALPHA); return; }   // the normal itself

    // max(dot, 0) IS THE WHOLE OF "A SURFACE FACING AWAY RECEIVES NOTHING". Without it the
    // back of the object is lit by a NEGATIVE amount, which subtracts and looks like a hole.
    float ndotl = max(dot(n, l), 0.0);
    vec3 lit = albedo * (AMBIENT + ndotl);
    oColour = vec4(lit, ALPHA);
}
