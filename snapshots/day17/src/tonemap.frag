#version 460
// THE LAST PASS. It reads the HDR colour and the blurred bright parts, adds them, applies an
// exposure and a curve, and writes the eight bits a screen can actually show.
//
// A TONE MAP IS NOT A CLAMP. Clamping throws away everything over 1.0 and leaves a flat white
// hole where the detail was. A curve compresses the top end instead, so a value of 8 and a
// value of 20 still differ after it -- which is what makes a bright scene look bright rather
// than broken.
layout(constant_id = 0) const int CURVE = 2;      // 0 none (clip), 1 Reinhard, 2 ACES
layout(constant_id = 1) const float EXPOSURE = 1.0;
layout(constant_id = 2) const float BLOOM = 0.55;
layout(constant_id = 3) const int SHOW_BLOOM = 0;

layout(set = 0, binding = 0) uniform sampler2D uHdr;
layout(set = 0, binding = 1) uniform sampler2D uBloom;

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 oColour;

// REINHARD: x / (1 + x). Every value maps into 0..1 and nothing is ever clipped, which is its
// whole appeal -- and it desaturates bright colours towards grey, which is its whole problem.
vec3 reinhard(vec3 x) { return x / (vec3(1.0) + x); }

// ACES, in the filmic approximation everybody uses. It is a curve fitted to the response of the
// Academy Color Encoding System, which is itself a fit to film -- so this is an approximation
// of a standard that approximates chemistry, and it looks better than either.
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(uHdr, vUV).rgb;
    vec3 bloom = texture(uBloom, vUV).rgb;
    if (SHOW_BLOOM == 1) { oColour = vec4(bloom, 1.0); return; }

    vec3 c = (hdr + bloom * BLOOM) * EXPOSURE;
    if (CURVE == 1) c = reinhard(c);
    else if (CURVE == 2) c = aces(c);
    // CURVE == 0 does nothing at all, and the attachment's UNORM format clamps for it --
    // silently, which is the point of that scene.

    // AND THEN sRGB, because a monitor is not linear. Everything above this line has been in
    // linear light, where adding two lights means adding two numbers; below it is the encoding
    // a display expects. Getting this backwards makes everything washed out or muddy, and it
    // is invisible until you compare.
    oColour = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
