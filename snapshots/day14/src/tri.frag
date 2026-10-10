#version 460
// SHADOWS, AND THE FOUR THINGS THAT GO WRONG.
//
// A SHADOW MAP IS A DEPTH BUFFER RENDERED FROM THE LIGHT. To find out whether a fragment is in
// shadow, transform it into the light's clip space and compare its depth against what the light
// recorded. Nearer means lit; further means something else was in the way.
//
// AND EVERY PROBLEM WITH IT IS A PRECISION PROBLEM. The map has a finite resolution, its depths
// have a finite number of bits, and the surface being tested is usually at a grazing angle to
// the light -- which is why bias, and why every value of bias is wrong somewhere.
layout(constant_id = 0) const float ALPHA = 1.0;
layout(constant_id = 1) const int MATERIAL_COUNT = 8;
layout(constant_id = 2) const int FALLOFF = 3;
layout(constant_id = 3) const float AMBIENT = 0.06;
layout(constant_id = 4) const float BIAS = 0.0022;
layout(constant_id = 5) const int PCF = 1;        // 0 one tap, 1 a 3x3 kernel
layout(constant_id = 6) const int MODE = 0;       // 0 shade, 1 no shadows, 2 the map itself
layout(constant_id = 7) const float TEXEL = 1.0 / 2048.0;

struct Lamp {
    vec4 position;
    vec4 colour;
};

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 eye;
    vec4 sun;              // xyz the direction TO the sun, w its strength
} cam;

layout(std430, set = 0, binding = 1) readonly buffer Lights {
    int  count;
    Lamp lamps[];
} lights;

// sampler2DShadow IS NOT sampler2D. The texture() call on it takes THREE coordinates -- u, v
// and a depth to compare against -- and returns how much of the comparison passed, in hardware,
// with the filtering applied to the RESULT of the comparison rather than to the depths. That
// is the difference between a soft edge and an average depth that means nothing.
layout(set = 0, binding = 2) uniform sampler2DShadow uShadow;

layout(set = 1, binding = 0) uniform sampler2D uMaterials[8];

layout(push_constant) uniform Push {
    mat4 model;
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

float attenuate(float dist, float range) {
    float inv = 1.0 / (1.0 + dist * dist);
    float t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv * t * t;
}

float sun_visibility(vec3 n, vec3 l) {
    if (MODE == 1) return 1.0;

    vec4 lp = cam.lightViewProj * vec4(vWorld, 1.0);
    // AN ORTHOGRAPHIC PROJECTION HAS w = 1, so this divide is a formality here and is written
    // anyway: a spot light's shadow map is perspective, and forgetting it there is a bug that
    // only appears on one kind of light.
    vec3 ndc = lp.xyz / lp.w;
    // VULKAN'S x AND y RUN -1 TO 1 AND ITS z RUNS 0 TO 1, so only two of the three are remapped.
    // In OpenGL all three needed it, and a copied shader silently halves the depth range.
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (ndc.z > 1.0 || ndc.z < 0.0) return 1.0;

    // SLOPE-SCALED BIAS. A surface at a grazing angle to the light covers more depth per shadow
    // texel, so it needs more bias -- and a surface facing the light needs almost none. One
    // constant for both is what makes every fixed bias wrong somewhere.
    float slope = clamp(1.0 - dot(n, l), 0.0, 1.0);
    float bias = BIAS * (0.35 + 1.65 * slope);
    float z = ndc.z - bias;

    if (PCF == 0) return texture(uShadow, vec3(uv, z));

    // A 3x3 KERNEL, WHICH IS NINE HARDWARE COMPARISONS AND NOT NINE DEPTH READS. Each tap is
    // already filtered across four texels by the comparison sampler, so a 3x3 of them is
    // effectively a 4x4 average of yes-or-no answers.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += texture(uShadow, vec3(uv + vec2(x, y) * TEXEL, z));
    return sum / 9.0;
}

void main() {
    vec3 albedo = texture(uMaterials[vMaterial % MATERIAL_COUNT], vUV).rgb * vColour;
    vec3 n = normalize(vNormal);

    if (MODE == 2) {
        // THE MAP ITSELF, as a picture. Depth is nearly all close to 1 on an orthographic
        // projection, so it is stretched to make anything visible at all.
        vec4 lp = cam.lightViewProj * vec4(vWorld, 1.0);
        vec3 ndc = lp.xyz / lp.w;
        float d = texture(uShadow, vec3(ndc.xy * 0.5 + 0.5, 0.0));
        oColour = vec4(vec3(d), ALPHA);
        return;
    }

    vec3 l = normalize(cam.sun.xyz);
    float vis = sun_visibility(n, l);
    vec3 sum = vec3(AMBIENT) + cam.sun.w * max(dot(n, l), 0.0) * vis;

    for (int i = 0; i < lights.count; ++i) {
        vec3  to = lights.lamps[i].position.xyz - vWorld;
        float d = length(to);
        if (d > lights.lamps[i].position.w) continue;
        vec3  ll = to / max(d, 1e-4);
        sum += lights.lamps[i].colour.rgb * lights.lamps[i].colour.w
             * max(dot(n, ll), 0.0) * attenuate(d, lights.lamps[i].position.w);
    }
    oColour = vec4(albedo * sum, ALPHA);
}
