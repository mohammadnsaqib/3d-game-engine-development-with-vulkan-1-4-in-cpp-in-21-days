#version 460
// COOK-TORRANCE, WHICH IS THREE FUNCTIONS AND A DIVISION.
//
// LAMBERT ANSWERS "HOW MUCH LIGHT ARRIVES". It says nothing about where the light GOES, which
// is why every surface in this book so far has looked like paper: paper is the one material for
// which scattering equally in all directions is correct.
//
// A MICROFACET MODEL ANSWERS THE OTHER HALF. A surface is a field of tiny mirrors; how much
// light reaches the eye is how many of them happen to point exactly halfway between the light
// and the eye. Three terms:
//
//   D   how many microfacets point that way          -- GGX, and roughness is its width
//   G   how many of those are shadowed by neighbours -- Smith, and it matters at grazing angles
//   F   how reflective the surface is at this angle  -- Schlick, and it is why everything
//                                                       becomes a mirror edge-on
//
// AND METALNESS IS NOT A SLIDER BETWEEN TWO MATERIALS. A metal has no diffuse component at all
// and tints its reflection; a dielectric has a white reflection and a coloured diffuse. The
// parameter chooses which of two behaviours the surface has, and values in between are a
// convenience for textures that blend.
layout(constant_id = 0) const float ALPHA = 1.0;
layout(constant_id = 1) const int MATERIAL_COUNT = 8;
layout(constant_id = 2) const int FALLOFF = 3;
layout(constant_id = 3) const float AMBIENT = 0.06;
layout(constant_id = 4) const float BIAS = 0.0022;
layout(constant_id = 5) const int PCF = 1;
// TEN VIEWS OF ONE FRAME. A renderer that can only show you its final answer is a renderer
// you cannot debug, and every one of these is two lines of shader.
//
//   0 the frame            5 roughness
//   1 overdraw             6 the shadow term alone
//   2 depth                7 specular only
//   3 world normals        8 diffuse only
//   4 albedo only          9 how many lamps reached this pixel
layout(constant_id = 6) const int MODE = 0;
layout(constant_id = 7) const float TEXEL = 1.0 / 2048.0;

const float PI = 3.14159265359;

struct Lamp { vec4 position; vec4 colour; };

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 eye;
    vec4 sun;
} cam;

layout(std430, set = 0, binding = 1) readonly buffer Lights {
    int  count;
    Lamp lamps[];
} lights;

layout(set = 0, binding = 2) uniform sampler2DShadow uShadow;
layout(set = 1, binding = 0) uniform sampler2D uMaterials[8];

layout(push_constant) uniform Push {
    mat4  model;
    vec4  tint;
    int   material;
    int   mode;
    float roughness;
    float metalness;
} pc;

layout(location = 0) in  vec3 vColour;
layout(location = 1) in  vec2 vUV;
layout(location = 2) flat in int vMaterial;
layout(location = 3) in  vec3 vNormal;
layout(location = 4) in  vec3 vWorld;
layout(location = 0) out vec4 oColour;

// D: THE DISTRIBUTION. GGX, also called Trowbridge-Reitz, and the reason it beat the older
// Blinn-Phong is its TAIL: a wide, slowly-fading skirt around a tight core, which is what real
// measured materials do and what a Gaussian does not.
float D_ggx(float ndoth, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// G: THE GEOMETRY TERM. At a grazing angle a microfacet is likely to be hidden behind its
// neighbours, and without this the edges of everything become impossibly bright.
float G_smith(float ndotv, float ndotl, float rough) {
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float gv = ndotv / (ndotv * (1.0 - k) + k);
    float gl = ndotl / (ndotl * (1.0 - k) + k);
    return gv * gl;
}

// F: FRESNEL, BY SCHLICK'S APPROXIMATION. Every surface becomes a mirror at a grazing angle --
// a road ahead of you at sunset, a sheet of paper held edge-on to a window. F0 is how
// reflective it is head-on: about 0.04 for anything non-metallic, and the albedo itself for a
// metal.
vec3 F_schlick(float cosine, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

float attenuate(float dist, float range) {
    float inv = 1.0 / (1.0 + dist * dist);
    float t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv * t * t;
}

float sun_visibility(vec3 n, vec3 l) {
    vec4 lp = cam.lightViewProj * vec4(vWorld, 1.0);
    vec3 ndc = lp.xyz / lp.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (ndc.z > 1.0 || ndc.z < 0.0) return 1.0;
    float slope = clamp(1.0 - dot(n, l), 0.0, 1.0);
    float z = ndc.z - BIAS * (0.35 + 1.65 * slope);
    if (PCF == 0) return texture(uShadow, vec3(uv, z));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += texture(uShadow, vec3(uv + vec2(x, y) * TEXEL, z));
    return sum / 9.0;
}

// ONE LIGHT'S CONTRIBUTION, and the whole model is in these fifteen lines.
// MODE 7 AND 8 SPLIT THE ANSWER IN TWO. A renderer that can show its specular and its
// diffuse separately is a renderer whose material bugs take minutes rather than afternoons.
vec3 shade(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo, float rough, float metal) {
    float ndotl = max(dot(n, l), 0.0);
    if (ndotl <= 0.0) return vec3(0.0);
    float ndotv = max(dot(n, v), 1e-4);
    vec3  h = normalize(v + l);            // the HALF VECTOR: the normal a mirror would need
    float ndoth = max(dot(n, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    if (MODE == 1) return albedo * radiance * ndotl;      // Lambert, for comparison

    // F0: 0.04 FOR EVERYTHING THAT IS NOT A METAL. That single number covers water, glass,
    // plastic, skin and stone to within a few per cent, which is the most useful constant in
    // real-time rendering.
    vec3 f0 = mix(vec3(0.04), albedo, metal);

    float D = D_ggx(ndoth, rough);
    float G = MODE == 3 ? 1.0 : G_smith(ndotv, ndotl, rough);
    vec3  F = MODE == 2 ? f0 : F_schlick(vdoth, f0);

    vec3 spec = (D * G) * F / max(4.0 * ndotv * ndotl, 1e-4);

    // ENERGY CONSERVATION: what is reflected is not also scattered. kd is what is LEFT after
    // the Fresnel term takes its share, and a metal keeps none of it.
    vec3 kd = (vec3(1.0) - F) * (1.0 - metal);
    if (MODE == 7) return spec * radiance * ndotl;
    if (MODE == 8) return kd * albedo / PI * radiance * ndotl;
    return (kd * albedo / PI + spec) * radiance * ndotl;
}

// A HEAT RAMP, because a number shown as brightness is a number nobody can read. Blue for
// one, green for four, red for eight and above -- and the same ramp is used for overdraw and
// for the lamp count so that the two pictures can be compared directly.
vec3 heat(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.6 * t - 0.6, 1.0 - 2.4 * abs(t - 0.5), 1.0 - 1.8 * t), 0.0, 1.0);
}

void main() {
    vec3 albedo = texture(uMaterials[vMaterial % MATERIAL_COUNT], vUV).rgb * vColour;
    vec3 n = normalize(vNormal);
    vec3 v = normalize(cam.eye.xyz - vWorld);
    float rough = clamp(pc.roughness, 0.03, 1.0);
    float metal = clamp(pc.metalness, 0.0, 1.0);

    // OVERDRAW IS COUNTED BY THE BLENDER, NOT BY THE SHADER. Every fragment that survives adds
    // an eighth, with the depth test off, so the colour channel IS the number of fragments that
    // reached this pixel. One pipeline field and one constant.
    if (MODE == 1) { oColour = vec4(0.125, 0.125, 0.125, 1.0); return; }
    if (MODE == 2) {
        float d = clamp(length(cam.eye.xyz - vWorld) / 40.0, 0.0, 1.0);
        oColour = vec4(vec3(1.0 - d), ALPHA);
        return;
    }
    if (MODE == 3) { oColour = vec4(n * 0.5 + 0.5, ALPHA); return; }
    if (MODE == 4) { oColour = vec4(albedo, ALPHA); return; }
    if (MODE == 5) { oColour = vec4(vec3(rough), ALPHA); return; }

    vec3 l = normalize(cam.sun.xyz);

    if (MODE == 6) { oColour = vec4(vec3(sun_visibility(n, l)), ALPHA); return; }

    // A METAL WITH NOTHING TO REFLECT IS BLACK. That is correct -- a metal has no diffuse
    // component at all, so with only point lights in the scene the brass rings come out nearly
    // invisible, which is what the first run of this chapter produced.
    //
    // THE STAND-IN IS AN AMBIENT THAT KNOWS ABOUT F0: a dielectric gets the usual flat term,
    // and a metal gets its own reflectance instead, brighter, standing in for a room it cannot
    // see. The real answer is an environment map, which is Day 17's territory.
    vec3 f0_amb = mix(vec3(0.04), albedo, metal);
    vec3 sum = albedo * AMBIENT * (1.0 - metal) + f0_amb * AMBIENT * 3.4 * metal;
    sum += shade(n, v, l, vec3(cam.sun.w) * sun_visibility(n, l), albedo, rough, metal);

    int reached = 0;
    for (int i = 0; i < lights.count; ++i) {
        vec3  to = lights.lamps[i].position.xyz - vWorld;
        float d = length(to);
        if (d > lights.lamps[i].position.w) continue;
        ++reached;
        vec3 radiance = lights.lamps[i].colour.rgb * lights.lamps[i].colour.w
                      * attenuate(d, lights.lamps[i].position.w);
        sum += shade(n, v, to / max(d, 1e-4), radiance, albedo, rough, metal);
    }
    if (MODE == 9) { oColour = vec4(heat(float(reached) / 6.0), ALPHA); return; }
    oColour = vec4(sum, ALPHA);
}
