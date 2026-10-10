#version 460
// THE FIRST SHADER IN THIS BOOK THAT IS GIVEN ITS GEOMETRY. For three days the corners came
// out of gl_VertexIndex and arithmetic, which is the right way to start and the wrong way to
// continue: a mesh somebody modelled cannot be computed from an integer.
//
// A location IS the contract. The pipeline's attribute descriptions say "location 0 is two
// floats starting at byte 0 of binding 0"; this says "location 0 is a vec2". Neither knows
// about the other, and nothing checks that they agree except the validation layer.
layout(constant_id = 0) const float ASPECT = 1.0;
layout(constant_id = 1) const float POINT_SIZE = 1.0;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColour;

layout(location = 0) out vec3 vColour;

void main() {
    vec2 p = inPosition;
    p.x /= ASPECT;                 // a circle stays a circle on a 16:9 image
    gl_Position = vec4(p, 0.0, 1.0);
    gl_PointSize = POINT_SIZE;     // ignored unless the topology is POINT_LIST
    vColour = inColour;
}
