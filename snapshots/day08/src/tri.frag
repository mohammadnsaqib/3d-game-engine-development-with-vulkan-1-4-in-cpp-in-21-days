#version 460
// THE FIRST DESCRIPTOR IN THIS BOOK. set 0, binding 0, and a combined image sampler -- which
// is Vulkan's name for an image and the rules for reading it, handed over as one object.
//
// THE SHADER DOES NOT KNOW WHICH IMAGE. It knows there is one at set 0, binding 0, with the
// shape sampler2D. What is actually there is decided by vkCmdBindDescriptorSets, at record
// time, and that indirection is the whole of tomorrow.
layout(constant_id = 0) const float ALPHA = 1.0;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) in  vec3 vColour;
layout(location = 1) in  vec2 vUV;
layout(location = 0) out vec4 oColour;

void main() {
    // texture() DOES NOT NAME A MIP LEVEL. The hardware works out how fast vUV is changing
    // between neighbouring pixels and picks one -- which is why a mip chain is a property of
    // the image and the level is never something you write down.
    vec4 t = texture(uTexture, vUV);
    oColour = vec4(t.rgb * vColour, ALPHA);
}
