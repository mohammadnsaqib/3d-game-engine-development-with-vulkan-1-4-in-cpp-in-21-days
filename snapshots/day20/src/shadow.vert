#version 460
// THE WHOLE OF THE SHADOW PASS. No fragment shader at all: the pass has a depth attachment and
// no colour attachment, so there is nothing for a fragment shader to write and the hardware's
// depth test does the entire job.
//
// A PASS WITH NO FRAGMENT SHADER IS THE FASTEST THING A GPU DOES. Vertices in, depth out, and
// the early-fragment-test hardware never calls anything.
layout(push_constant) uniform Push {
    mat4 lightViewProjModel;
} pc;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = pc.lightViewProjModel * vec4(inPosition, 1.0);
}
