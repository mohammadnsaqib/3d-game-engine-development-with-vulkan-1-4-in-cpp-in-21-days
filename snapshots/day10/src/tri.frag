#version 460
// SET 1 IS THE MATERIALS, AND IT IS AN ARRAY OF EIGHT. One descriptor set holding eight
// combined image samplers, indexed by a number that arrives in a push constant -- which is
// how sixty-four objects with eight materials become sixty-four draws and ONE bind.
//
// nonuniformEXT WOULD BE NEEDED IF THE INDEX VARIED WITHIN A DRAW. It does not here: the
// index is a push constant, constant across the whole draw, which is the easy case and the
// one that works without any extension at all.
layout(constant_id = 0) const float ALPHA = 1.0;
layout(constant_id = 1) const int MATERIAL_COUNT = 8;

layout(set = 1, binding = 0) uniform sampler2D uMaterials[8];

layout(location = 0) in  vec3 vColour;
layout(location = 1) in  vec2 vUV;
layout(location = 2) flat in int vMaterial;
layout(location = 0) out vec4 oColour;

void main() {
    vec4 t = texture(uMaterials[vMaterial % MATERIAL_COUNT], vUV);
    oColour = vec4(t.rgb * vColour, ALPHA);
}
