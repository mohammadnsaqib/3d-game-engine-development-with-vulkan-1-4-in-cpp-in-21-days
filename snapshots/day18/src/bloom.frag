#version 460
// TWO JOBS IN ONE SHADER, chosen by a specialization constant: a bright pass that keeps only
// what is over the threshold, and one axis of a separable Gaussian blur.
//
// SEPARABLE MEANS TWO PASSES OF N TAPS RATHER THAN ONE OF N SQUARED. A 13x13 blur is 169 taps
// done directly and 26 done as two passes, and the result is identical because a 2D Gaussian
// is the product of two 1D ones. It is the single best-value identity in graphics.
layout(constant_id = 0) const int STAGE = 0;      // 0 bright pass, 1 horizontal, 2 vertical
layout(constant_id = 1) const float THRESHOLD = 1.0;
layout(constant_id = 2) const float TEXEL_X = 1.0 / 960.0;
layout(constant_id = 3) const float TEXEL_Y = 1.0 / 540.0;

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 oColour;

// A NINE-TAP GAUSSIAN. The weights are a row of Pascal's triangle, normalised, and they are
// written out rather than computed because nine constants are cheaper than nine exp() calls
// and because a reader can check them against a table.
const float W[5] = float[5](0.2270270270, 0.1945945946, 0.1216216216,
                            0.0540540541, 0.0162162162);

void main() {
    if (STAGE == 0) {
        vec3 c = texture(uSource, vUV).rgb;
        // WHAT IS "BRIGHT" IS A CHOICE. The usual measure is luminance, weighted by how
        // sensitive the eye is to each channel -- so a saturated blue at 1.2 blooms less than
        // a white at 1.2, which is what you want and what max(r,g,b) would not give.
        float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
        float keep = max(lum - THRESHOLD, 0.0) / max(lum, 1e-4);
        oColour = vec4(c * keep, 1.0);
        return;
    }
    vec2 step = STAGE == 1 ? vec2(TEXEL_X, 0.0) : vec2(0.0, TEXEL_Y);
    vec3 sum = texture(uSource, vUV).rgb * W[0];
    for (int i = 1; i < 5; ++i) {
        sum += texture(uSource, vUV + step * float(i)).rgb * W[i];
        sum += texture(uSource, vUV - step * float(i)).rgb * W[i];
    }
    oColour = vec4(sum, 1.0);
}
