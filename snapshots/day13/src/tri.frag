#version 460
// LAMPS, AND THE FOUR ANSWERS TO "HOW FAST DOES LIGHT RUN OUT".
//
// A DIRECTIONAL LIGHT HAS NO POSITION and therefore no distance. A point light has both, and
// the difference is the entire chapter: the direction to the light is recomputed per fragment,
// and the amount arriving falls off with distance.
//
// THE LIGHTS ARE A STORAGE BUFFER, not constants. Yesterday the sun was compiled into the
// shader and every scene was a different pipeline; today there are up to 64 lamps in a buffer
// and a count, and changing all of them is a memcpy.
layout(constant_id = 0) const float ALPHA = 1.0;
layout(constant_id = 1) const int MATERIAL_COUNT = 8;
layout(constant_id = 2) const int FALLOFF = 0;      // 0 inverse square, 1 none, 2 linear,
                                                    // 3 inverse square with a window
layout(constant_id = 3) const float AMBIENT = 0.05;

struct Lamp {
    vec4 position;      // xyz, and w is the range
    vec4 colour;        // rgb, and w is the intensity
};

layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 eye;
} cam;

// A STORAGE BUFFER IS READ-ONLY HERE AND DOES NOT HAVE TO BE. That is the difference from a
// uniform buffer: a uniform buffer is small, fast and immutable to the shader; a storage buffer
// can be megabytes, can be indexed by anything, and can be WRITTEN. Day 18 writes one.
layout(std430, set = 0, binding = 1) readonly buffer Lights {
    int  count;
    Lamp lamps[];
} lights;

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
    if (FALLOFF == 1) return 1.0;                              // none: a lamp with no distance
    if (FALLOFF == 2) return max(1.0 - dist / range, 0.0);     // linear: cheap, and wrong
    // INVERSE SQUARE IS THE PHYSICAL ANSWER: the same light spread over the surface of a
    // sphere whose area is 4 pi r squared. The +1 in the denominator is the standard dodge
    // that stops it reaching infinity at distance zero.
    float inv = 1.0 / (1.0 + dist * dist);
    if (FALLOFF == 0) return inv;
    // AND THE GAME ANSWER: inverse square multiplied by a window that reaches exactly zero at
    // the range, so a lamp can be skipped entirely beyond it without a visible edge.
    float t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv * t * t;
}

void main() {
    vec3 albedo = texture(uMaterials[vMaterial % MATERIAL_COUNT], vUV).rgb * vColour;
    vec3 n = normalize(vNormal);

    if (pc.mode == 1) { oColour = vec4(albedo, ALPHA); return; }
    if (pc.mode == 2) { oColour = vec4(n * 0.5 + 0.5, ALPHA); return; }

    vec3 sum = vec3(AMBIENT);
    for (int i = 0; i < lights.count; ++i) {
        vec3  to = lights.lamps[i].position.xyz - vWorld;
        float d = length(to);
        // A HARD RANGE TEST IS A VISIBLE CIRCLE unless the falloff already reaches zero there,
        // which is exactly what mode 3 is for.
        if (d > lights.lamps[i].position.w) continue;
        vec3  l = to / max(d, 1e-4);
        float ndotl = max(dot(n, l), 0.0);
        sum += lights.lamps[i].colour.rgb * lights.lamps[i].colour.w
             * ndotl * attenuate(d, lights.lamps[i].position.w);
    }
    oColour = vec4(albedo * sum, ALPHA);
}
