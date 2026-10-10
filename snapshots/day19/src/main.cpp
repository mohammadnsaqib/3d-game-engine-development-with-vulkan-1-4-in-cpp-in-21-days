// =======================================================================================
// MERIDIAN - Day 17.  The lamp gets hot.
//
// FOR SIXTEEN DAYS THE COLOUR BUFFER HAS BEEN EIGHT BITS PER CHANNEL. Day 12 said a patch under
// three bright lamps produces a value well over 1.0 and that writing it to an R8G8B8A8_UNORM
// attachment clips it, silently, to white. Today it stops.
//
// FOUR PASSES NOW, which is what Day 16's chapter was preparing for:
//
//   1  shadow      depth from the sun                       D32_SFLOAT, 2048 square
//   2  main        the scene, in linear light, unclamped    R16G16B16A16_SFLOAT
//   3  bloom       bright pass, then blur across, then down  half resolution
//   4  tone map    add, expose, curve, encode                R8G8B8A8_UNORM
//
// A TONE MAP IS NOT A CLAMP. Clamping throws away everything over 1.0 and leaves a flat white
// hole. A curve compresses the top end so that 8 and 20 still differ afterwards, which is what
// makes a bright scene look bright rather than broken.
//
// AND THE LAST THREE PASSES DRAW ONE TRIANGLE EACH, with no vertex buffer -- three corners out
// of gl_VertexIndex, which is where Day 1 started and is the one place in a finished engine
// where it is still the right answer.
//
//   meridian.exe               renders every scene below and exits
//   meridian.exe --scene NAME  renders one of them
//   meridian.exe --window      open a window and present until Esc
//   meridian.exe --provoke     break a rule on purpose, and print what validation says
// =======================================================================================
#define WIN32_LEAN_AND_MEAN
// VK_USE_PLATFORM_WIN32_KHR IS WHAT MAKES VkWin32SurfaceCreateInfoKHR EXIST. The Vulkan header
// declares every platform's surface types behind a define, because a header that declared all
// of them would need every platform s own headers to be present. This is the one line in the
// book that says which operating system it is.
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <wincodec.h>
#include <vulkan/vulkan.h>

#include "vk_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "spirv.h"

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

// THE CLEAR COLOUR, and it is the same night blue the whole book is lit against.
static const float NIGHT[4] = { 0.055f, 0.066f, 0.098f, 1.0f };

static const uint32_t RENDER_W = 1920, RENDER_H = 1080;

// ---- THE GEOMETRY ---------------------------------------------------------------------
// A VERTEX HAS THREE COORDINATES NOW, and that one change is what makes everything below
// necessary. Twenty-four bytes: a position in the object's own space and a colour.
struct Vertex {
    float x, y, z;     // location 0, VK_FORMAT_R32G32B32_SFLOAT, offset 0
    float r, g, b;     // location 1, VK_FORMAT_R32G32B32_SFLOAT, offset 12
    float u, v;        // location 2, VK_FORMAT_R32G32_SFLOAT,    offset 24
    float nx, ny, nz;  // location 3, VK_FORMAT_R32G32B32_SFLOAT, offset 32
};

// A HUE WHEEL, back from Day 4, because the texture needs colours that say where on it a
// texel came from. A grey checkerboard would show the filtering and hide the wrapping.
static void hue(float t, float& r, float& g, float& b) {
    const float h = t * 6.0f;
    const int   i = int(h) % 6;
    const float f = h - float(int(h));
    switch (i) {
        case 0: r = 1.0f;     g = f;        b = 0.0f;     break;
        case 1: r = 1.0f - f; g = 1.0f;     b = 0.0f;     break;
        case 2: r = 0.0f;     g = 1.0f;     b = f;        break;
        case 3: r = 0.0f;     g = 1.0f - f; b = 1.0f;     break;
        case 4: r = f;        g = 0.0f;     b = 1.0f;     break;
        default: r = 1.0f;    g = 0.0f;     b = 1.0f - f; break;
    }
}


// THE TEXTURE, GENERATED RATHER THAN LOADED. A file would need a decoder, a path, and an
// error case, none of which is about Vulkan. This is 512x512 of deliberately awkward content:
// a coloured tile grid with a ring in each tile and a one-texel line across it, so that
// minification has something to destroy and a mip chain has something to save.
static const uint32_t TEX_SIZE = 256;
static const int MATERIAL_COUNT = 8;

// EIGHT MATERIALS OUT OF ONE GENERATOR. The parameter is what makes them different, and the
// point of eight is that the engine has to CHOOSE one per object -- which is the whole reason
// descriptor sets exist.
static void build_texture(std::vector<uint8_t>& px, int material) {
    px.resize(size_t(TEX_SIZE) * TEX_SIZE * 4);
    const uint32_t tile = 32u << (material % 3);
    for (uint32_t y = 0; y < TEX_SIZE; ++y)
        for (uint32_t x = 0; x < TEX_SIZE; ++x) {
            const uint32_t tx = x / tile, ty = y / tile;
            float r = 0, g = 0, b = 0;
            hue(float((tx * 5 + ty * 3 + material * 3) % 8) / 8.0f, r, g, b);
            const float shade = ((tx + ty) & 1) ? 1.0f : 0.55f;
            r *= shade; g *= shade; b *= shade;
            // a ring inside each tile
            const float half = float(tile) * 0.5f - 0.5f;
            const float cx = float(x % tile) - half, cy = float(y % tile) - half;
            const float d = std::sqrt(cx * cx + cy * cy);
            if (d > half * 0.58f && d < half * 0.78f) { r = 0.06f; g = 0.07f; b = 0.11f; }
            // AND A ONE-TEXEL LINE, which is what aliases. A feature exactly one texel wide
            // cannot survive being halved, and that is the entire argument for mip maps.
            if ((x % 16) == 0 || (y % 16) == 0) {
                r = 0.96f - 0.10f * float(material);
                g = 0.92f;
                b = 0.86f + 0.02f * float(material);
            }
            const size_t o = (size_t(y) * TEX_SIZE + x) * 4;
            px[o + 0] = uint8_t(r * 255.0f);
            px[o + 1] = uint8_t(g * 255.0f);
            px[o + 2] = uint8_t(b * 255.0f);
            px[o + 3] = 255;
        }
}

// A GROUND PLANE, so that minification has somewhere to happen. A cube seen from four units
// away magnifies its texture; a floor running to the horizon does the opposite, and every
// sampler setting below is about one of those two directions.
static void build_plane(std::vector<Vertex>& v, std::vector<uint32_t>& idx, float half,
                        float uv) {
    v.push_back({ -half, 0.0f, -half, 1, 1, 1, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f });
    v.push_back({  half, 0.0f, -half, 1, 1, 1,   uv, 0.0f, 0.0f, 1.0f, 0.0f });
    v.push_back({  half, 0.0f,  half, 1, 1, 1,   uv,   uv, 0.0f, 1.0f, 0.0f });
    v.push_back({ -half, 0.0f,  half, 1, 1, 1, 0.0f,   uv, 0.0f, 1.0f, 0.0f });
    const uint32_t q[6] = { 0, 1, 2, 0, 2, 3 };
    idx.assign(q, q + 6);
}

// ---- FOUR BY FOUR, COLUMN MAJOR, AND WRITTEN OUT RATHER THAN LINKED IN --------------------
// GLM would do all of this in a header. It is fifty lines here because a reader who has never
// multiplied two matrices by hand should see what the multiplication IS before importing one.
//
// COLUMN MAJOR because that is what GLSL expects: m[column][row], and element c*4+r. Getting
// this backwards transposes every matrix in the program and produces a picture that is almost
// right, which is far worse than one that is obviously wrong.
struct Mat4 { float m[16]; };

static Mat4 identity() {
    Mat4 r = {};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

// a * b, meaning "apply b, then a" -- the order everybody gets wrong once.
static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r = {};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
    return r;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x; r.m[13] = y; r.m[14] = z;
    return r;
}

static Mat4 rotate_y(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
    return r;
}

// ROTATE ABOUT Z, which this engine has not needed for twelve days and needs exactly once:
// to show what a camera with a roll looks like, and why nobody wants one.
static Mat4 rotate_z(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c;
    return r;
}

static Mat4 rotate_x(float a) {
    Mat4 r = identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
    return r;
}

// THE VIEW MATRIX IS THE CAMERA'S INVERSE. A camera at (0, 0, 4) looking at the origin does not
// move the camera -- it moves the WORLD four units towards the viewer. Everything here is
// built from that one idea.
static Mat4 look_at(float ex, float ey, float ez) {
    float fx = -ex, fy = -ey, fz = -ez;                    // forward: eye towards origin
    const float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0.0f - fz * 1.0f, sy = fz * 0.0f - fx * 0.0f, sz = fx * 1.0f - fy * 0.0f;
    const float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
    sx /= sl; sy /= sl; sz /= sl;
    const float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    Mat4 r = identity();
    r.m[0] = sx; r.m[4] = sy; r.m[8]  = sz;
    r.m[1] = ux; r.m[5] = uy; r.m[9]  = uz;
    r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
    r.m[12] = -(sx * ex + sy * ey + sz * ez);
    r.m[13] = -(ux * ex + uy * ey + uz * ez);
    r.m[14] = fx * ex + fy * ey + fz * ez;
    return r;
}

// THE PROJECTION, AND THE ONE NEGATED NUMBER THIS BOOK HAS BEEN WAITING FOR.
//
//   m[5] = -cot(fov/2)   the y flip, because Vulkan's y points DOWN
//   depth 0..1, not -1..1 -- OpenGL's range is the odd one out, not Vulkan's
static Mat4 perspective(float fov_y, float aspect, float znear, float zfar, bool flip) {
    const float f = 1.0f / std::tan(fov_y * 0.5f);
    Mat4 r = {};
    r.m[0] = f / aspect;
    r.m[5] = flip ? -f : f;
    r.m[10] = zfar / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (znear * zfar) / (znear - zfar);
    return r;
}

// NO PERSPECTIVE DIVIDE AT ALL: parallel lines stay parallel, and distance stops mattering.
static Mat4 ortho(float half, float aspect, float znear, float zfar, bool flip) {
    Mat4 r = identity();
    r.m[0] = 1.0f / (half * aspect);
    r.m[5] = (flip ? -1.0f : 1.0f) / half;
    r.m[10] = 1.0f / (znear - zfar);
    r.m[14] = znear / (znear - zfar);
    return r;
}

// ---- A MESH IS PARTS, AND A PART IS THREE INTEGERS --------------------------------------
// firstIndex and indexCount say which slice of the index buffer belongs to this part.
// vertexOffset is added to every index in that slice before the vertex is fetched, which is
// what lets a generator write its indices from zero and still land in the right place.
struct Part {
    const char* name;
    uint32_t    first_index, index_count;
    int32_t     vertex_offset;
    int         material;
    float       ox, oy, oz;     // where the part sits inside the object
    float       roughness;      // 0.03 is a mirror, 1.0 is chalk
    float       metalness;      // 0 or 1, and the values between are for textures
};

// A BUILDER, so that each generator can be written as if it were alone in the world.
struct MeshBuilder {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    std::vector<Part>     parts;

    void begin() { part_first_ = uint32_t(indices.size()); part_base_ = int32_t(vertices.size()); }

    void end(const char* name, int material, float ox = 0, float oy = 0, float oz = 0,
             float roughness = 0.45f, float metalness = 0.0f) {
        parts.push_back({ name, part_first_, uint32_t(indices.size()) - part_first_,
                          part_base_, material, ox, oy, oz, roughness, metalness });
    }

    void tri(uint32_t a, uint32_t b, uint32_t c) {
        indices.push_back(a); indices.push_back(b); indices.push_back(c);
    }
    void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { tri(a, b, c); tri(a, c, d); }

    uint32_t here() const { return uint32_t(vertices.size()) - uint32_t(part_base_); }

  private:
    uint32_t part_first_ = 0;
    int32_t  part_base_ = 0;
};

// ---- FOUR GENERATORS ---------------------------------------------------------------------
// A UV SPHERE: rings of latitude, segments of longitude, and two poles that are the reason
// every sphere generator is uglier than it looks. The pole vertices are duplicated per
// longitude because each of them needs a different texture coordinate -- Day 6's argument,
// arriving again.
static void gen_sphere(MeshBuilder& m, float radius, int rings, int segs,
                       float r, float g, float b) {
    const float PI = 3.14159265358979f;
    const uint32_t base = m.here();
    for (int y = 0; y <= rings; ++y) {
        const float v = float(y) / float(rings);
        const float phi = v * PI;
        for (int x = 0; x <= segs; ++x) {
            const float u = float(x) / float(segs);
            const float theta = u * 2.0f * PI;
            // A SPHERE'S NORMAL IS FREE: it is the direction from the centre, which is the
            // position divided by the radius. Most shapes are not this kind.
            const float nx = std::sin(phi) * std::cos(theta);
            const float ny = std::cos(phi);
            const float nz = std::sin(phi) * std::sin(theta);
            m.vertices.push_back({ radius * nx, radius * ny, radius * nz,
                                   r, g, b, u * 4.0f, v * 2.0f, nx, ny, nz });
        }
    }
    for (int y = 0; y < rings; ++y)
        for (int x = 0; x < segs; ++x) {
            const uint32_t i0 = base + uint32_t(y * (segs + 1) + x);
            m.quad(i0, i0 + uint32_t(segs) + 1, i0 + uint32_t(segs) + 2, i0 + 1);
        }
}

// A TORUS: a circle swept around another circle. Three of these, at three angles, are what
// makes an armillary sphere look like one.
static void gen_torus(MeshBuilder& m, float major, float minor, int around, int through,
                      int axis, float r, float g, float b) {
    const float PI = 3.14159265358979f;
    const uint32_t base = m.here();
    for (int i = 0; i <= around; ++i) {
        const float u = float(i) / float(around), a = u * 2.0f * PI;
        const float ca = std::cos(a), sa = std::sin(a);
        for (int j = 0; j <= through; ++j) {
            const float v = float(j) / float(through), t = v * 2.0f * PI;
            const float ct = std::cos(t), st = std::sin(t);
            const float rad = major + minor * ct;
            float px = rad * ca, py = minor * st, pz = rad * sa;
            // A TORUS'S NORMAL is the direction from the centre of the TUBE, not from the
            // centre of the torus -- which is the difference between a doughnut and a sphere.
            float nx = ct * ca, ny = st, nz = ct * sa;
            if (axis == 1) {
                const float k = py; py = px; px = k;                       // stand it up
                const float kn = ny; ny = nx; nx = kn;
            }
            if (axis == 2) {
                const float k = py; py = pz; pz = k;                       // and the third way
                const float kn = ny; ny = nz; nz = kn;
            }
            m.vertices.push_back({ px, py, pz, r, g, b, u * 8.0f, v, nx, ny, nz });
        }
    }
    for (int i = 0; i < around; ++i)
        for (int j = 0; j < through; ++j) {
            const uint32_t i0 = base + uint32_t(i * (through + 1) + j);
            m.quad(i0, i0 + 1, i0 + uint32_t(through) + 2, i0 + uint32_t(through) + 1);
        }
}

// A CYLINDER, for the pointer through the poles and the legs of the stand.
static void gen_cylinder(MeshBuilder& m, float radius, float half_height, int segs,
                         float r, float g, float b) {
    const float PI = 3.14159265358979f;
    const uint32_t base = m.here();
    for (int i = 0; i <= segs; ++i) {
        const float u = float(i) / float(segs), a = u * 2.0f * PI;
        const float ca = std::cos(a), sa = std::sin(a);
        // A CYLINDER'S SIDE FACES OUTWARD AND NOWHERE ELSE: no y component at all, which is
        // why a cylinder lit from directly above goes dark and a sphere does not.
        m.vertices.push_back({ radius * ca, -half_height, radius * sa, r, g, b, u * 4.0f, 0.0f,
                               ca, 0.0f, sa });
        m.vertices.push_back({ radius * ca,  half_height, radius * sa, r, g, b, u * 4.0f, 1.0f,
                               ca, 0.0f, sa });
    }
    for (int i = 0; i < segs; ++i) {
        const uint32_t i0 = base + uint32_t(i * 2);
        m.quad(i0, i0 + 2, i0 + 3, i0 + 1);
    }
}

// AND A DISC, which closes the stand and is the only generator here with a centre vertex.
static void gen_disc(MeshBuilder& m, float radius, float y, int segs,
                     float r, float g, float b) {
    const float PI = 3.14159265358979f;
    const uint32_t base = m.here();
    m.vertices.push_back({ 0.0f, y, 0.0f, r, g, b, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f });
    for (int i = 0; i <= segs; ++i) {
        const float u = float(i) / float(segs), a = u * 2.0f * PI;
        m.vertices.push_back({ radius * std::cos(a), y, radius * std::sin(a), r, g, b,
                               0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a),
                               0.0f, 1.0f, 0.0f });
    }
    for (int i = 0; i < segs; ++i)
        m.tri(base, base + uint32_t(i) + 1, base + uint32_t(i) + 2);
}

// ---- FLAT SHADING IS A MESH, NOT A SETTING ------------------------------------------------
// Every triangle gets its own three vertices, each carrying the FACE's normal -- which is the
// cross product of two of its edges. The index buffer becomes 0, 1, 2, 3, 4, 5 ... and the
// vertex count becomes three times the triangle count.
//
// THAT IS DAY 6'S RULE COSTING REAL MEMORY. A vertex is every attribute a face needs at a
// point, and a corner shared by three faces with three different normals is three vertices.
static void flatten(MeshBuilder& m) {
    std::vector<Vertex> out;
    out.reserve(m.indices.size());
    std::vector<uint32_t> idx;
    idx.reserve(m.indices.size());

    for (Part& pt : m.parts) {
        const uint32_t first = uint32_t(out.size());
        for (uint32_t i = 0; i < pt.index_count; i += 3) {
            Vertex a = m.vertices[size_t(int32_t(m.indices[pt.first_index + i + 0])
                                         + pt.vertex_offset)];
            Vertex b = m.vertices[size_t(int32_t(m.indices[pt.first_index + i + 1])
                                         + pt.vertex_offset)];
            Vertex c = m.vertices[size_t(int32_t(m.indices[pt.first_index + i + 2])
                                         + pt.vertex_offset)];
            const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
            const float wx = c.x - a.x, wy = c.y - a.y, wz = c.z - a.z;
            float nx = uy * wz - uz * wy;
            float ny = uz * wx - ux * wz;
            float nz = ux * wy - uy * wx;
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            // A DEGENERATE TRIANGLE HAS NO NORMAL, and a sphere's poles are full of them.
            // Dividing by zero here puts a NaN in the vertex buffer, and a NaN normal makes
            // one triangle black on some drivers and the whole object black on others.
            if (len > 1e-9f) { nx /= len; ny /= len; nz /= len; }
            else { nx = 0.0f; ny = 1.0f; nz = 0.0f; }
            a.nx = b.nx = c.nx = nx;
            a.ny = b.ny = c.ny = ny;
            a.nz = b.nz = c.nz = nz;
            idx.push_back(uint32_t(out.size()) - first);
            out.push_back(a);
            idx.push_back(uint32_t(out.size()) - first);
            out.push_back(b);
            idx.push_back(uint32_t(out.size()) - first);
            out.push_back(c);
        }
        pt.first_index = first;
        pt.index_count = uint32_t(out.size()) - first;
        pt.vertex_offset = int32_t(first);
    }
    m.vertices.swap(out);
    m.indices.swap(idx);
}

// ---- THE MERIDIAN ------------------------------------------------------------------------
// Nine parts, and the only thing that distinguishes this from a scene file is that it is
// arithmetic. Day 20 writes one of these to disk and reads it back.
static void build_meridian(MeshBuilder& m, int sphere_rings, int sphere_segs) {
    m.begin();
    gen_sphere(m, 1.00f, sphere_rings, sphere_segs, 0.95f, 0.72f, 0.28f);
    m.end("globe", 2, 0, 0, 0, 0.55f, 0.0f);

    m.begin();
    gen_torus(m, 1.55f, 0.055f, 96, 10, 0, 0.85f, 0.88f, 0.95f);
    m.end("equator", 5, 0, 0, 0, 0.18f, 1.0f);

    m.begin();
    gen_torus(m, 1.72f, 0.055f, 96, 10, 1, 0.55f, 0.80f, 1.00f);
    m.end("colure", 1, 0, 0, 0, 0.22f, 1.0f);

    m.begin();
    gen_torus(m, 1.89f, 0.055f, 96, 10, 2, 1.00f, 0.45f, 0.35f);
    m.end("tropic", 7, 0, 0, 0, 0.30f, 1.0f);

    m.begin();
    gen_cylinder(m, 0.045f, 2.25f, 24, 0.90f, 0.90f, 0.92f);
    m.end("axis", 5, 0, 0, 0, 0.12f, 1.0f);

    m.begin();
    gen_torus(m, 2.15f, 0.09f, 128, 12, 0, 0.75f, 0.62f, 0.42f);
    m.end("horizon", 4, 0, 0, 0, 0.35f, 1.0f);

    m.begin();
    gen_cylinder(m, 0.10f, 1.30f, 20, 0.62f, 0.50f, 0.38f);
    m.end("column", 4, 0, -3.50f, 0, 0.60f, 0.0f);

    m.begin();
    gen_disc(m, 1.20f, 0.0f, 48, 0.55f, 0.45f, 0.34f);
    m.end("plinth", 6, 0, -4.80f, 0, 0.75f, 0.0f);

    m.begin();
    gen_torus(m, 1.05f, 0.14f, 64, 10, 0, 0.68f, 0.56f, 0.40f);
    m.end("rim", 4, 0, -4.66f, 0, 0.28f, 1.0f);
}

// A CUBE WITH TWENTY-FOUR VERTICES, not eight. Eight would share every corner between three
// faces, and a shared corner cannot have three different colours -- which is the same reason
// Day 11's normals will force the count up again. A vertex is not a point in space; it is
// every attribute a face needs at that point.
static void build_cube(std::vector<Vertex>& v, std::vector<uint32_t>& idx) {
    struct Face { float nx, ny, nz; float r, g, b; };
    static const Face F[6] = {
        {  0,  0,  1, 1.00f, 0.36f, 0.18f },   // +z  ember
        {  0,  0, -1, 0.20f, 0.55f, 1.00f },   // -z  sky
        {  1,  0,  0, 0.98f, 0.78f, 0.25f },   // +x  brass
        { -1,  0,  0, 0.25f, 0.85f, 0.52f },   // -x  moss
        {  0,  1,  0, 0.85f, 0.30f, 0.75f },   // +y  orchid
        {  0, -1,  0, 0.55f, 0.58f, 0.68f },   // -y  stone
    };
    for (int f = 0; f < 6; ++f) {
        const Face& fa = F[f];
        // two vectors across the face, from its normal
        float ax = fa.ny, ay = fa.nz, az = fa.nx;
        const float bx = fa.ny * az - fa.nz * ay;
        const float by = fa.nz * ax - fa.nx * az;
        const float bz = fa.nx * ay - fa.ny * ax;
        const uint32_t base = uint32_t(v.size());
        const float sx[4] = { -1, 1, 1, -1 }, sy[4] = { -1, -1, 1, 1 };
        const float tu[4] = { 0, 1, 1, 0 }, tv[4] = { 0, 0, 1, 1 };
        for (int i = 0; i < 4; ++i)
            v.push_back({ fa.nx + ax * sx[i] + bx * sy[i],
                          fa.ny + ay * sx[i] + by * sy[i],
                          fa.nz + az * sx[i] + bz * sy[i],
                          fa.r, fa.g, fa.b, tu[i], tv[i] });
        const uint32_t q[6] = { base, base + 1, base + 2, base, base + 2, base + 3 };
        idx.insert(idx.end(), q, q + 6);
    }
}

// ---- A CAMERA IS A PLACE AND TWO ANGLES ---------------------------------------------------
// YAW turns left and right about the world's y axis. PITCH looks up and down about the
// camera's own x axis. There is no roll, because a first-person camera does not have one and
// the scene that adds one shows what it costs.
struct Camera {
    float x = 0.0f, y = 3.4f, z = 11.0f;
    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
    float fov_deg = 45.0f;
};

// THE CAMERA'S OWN PLACEMENT: where it is and which way it is turned, as a matrix that would
// position a MODEL there. This is the matrix nobody uses directly and everybody needs to think
// about, because the view matrix is its inverse.
static Mat4 camera_placement(const Camera& cam) {
    return mul(translate(cam.x, cam.y, cam.z),
               mul(rotate_y(cam.yaw), mul(rotate_x(cam.pitch), rotate_z(cam.roll))));
}

// AND THE VIEW MATRIX, BUILT DIRECTLY RATHER THAN INVERTED.
//
//   a rotation's inverse is its TRANSPOSE          -- R(-a) = R(a) transposed
//   a translation's inverse is its NEGATION        -- T(-p)
//   and (T R)^-1 = R^-1 T^-1, which is the order below, backwards
//
// A general 4x4 inversion would give the same answer, cost about forty times as much, and lose
// precision every frame. Knowing what the matrix IS is worth more than any library.
static Mat4 camera_view(const Camera& cam) {
    return mul(rotate_z(-cam.roll),
               mul(rotate_x(-cam.pitch),
                   mul(rotate_y(-cam.yaw), translate(-cam.x, -cam.y, -cam.z))));
}

// WHICH WAY IS FORWARD. The camera looks down its own -z, so forward in WORLD space is the
// third column of its placement, negated -- which is arithmetic rather than a second set of
// trigonometry to keep in step with the first.
static void camera_axes(const Camera& cam, float* fwd, float* right) {
    const Mat4 m = camera_placement(cam);
    fwd[0] = -m.m[8];  fwd[1] = -m.m[9];  fwd[2] = -m.m[10];
    right[0] = m.m[0]; right[1] = m.m[1]; right[2] = m.m[2];
}

// HOW FAR FROM THE IDENTITY. The whole claim of this chapter, as a number.
static float off_identity(const Mat4& a) {
    float worst = 0.0f;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            const float want = (c == r) ? 1.0f : 0.0f;
            const float got = a.m[c * 4 + r];
            const float e = got > want ? got - want : want - got;
            if (e > worst) worst = e;
        }
    return worst;
}

// ---- THE LAMPS ---------------------------------------------------------------------------
// std430 IS WHY THIS STRUCT IS TWO vec4s. A vec3 in std430 is aligned to sixteen bytes and
// padded to sixteen, so a struct of two vec3s and two floats occupies exactly the same space as
// two vec4s and reads worse. Packing the range into position.w and the intensity into colour.w
// is not a trick -- it is what the layout rules were going to do anyway.
struct Lamp {
    float px, py, pz, range;
    float r, g, b, intensity;
};
static const int MAX_LAMPS = 64;

// ---- THE DUST ----------------------------------------------------------------------------
// THIRTY-TWO BYTES, AND THE SAME LAYOUT ARGUMENT THE LAMPS MADE. Two vec4s, because std430
// would pad two vec3s and two floats to exactly this and read worse. It is declared here in
// C++ and again in dust.comp in GLSL, and the ONLY thing keeping the two in agreement is that
// somebody wrote them to agree -- which is the standing cost of a buffer two languages share.
struct Mote {
    float x, y, z, size;
    float r, g, b, life;
};
// THE BUFFER IS ALLOCATED ONCE, AT THE LARGEST COUNT ANY SCENE ASKS FOR. Resizing a device
// -local buffer means a queue idle, a destroy and an allocate, and the whole point of today is
// that the dust costs nothing per frame.
static const uint32_t MAX_MOTES = 160000;
// THIRTY-TWO, BECAUSE THAT IS WHAT THE MEASUREMENT SAID. Every tutorial writes 64 and this
// engine did too, until the benchmark below timed five sizes on this card and 32 came out a
// third faster. It is also exactly the subgroup size, which is the one number on the card that
// a workgroup size has any business being a multiple of.
static const uint32_t DUST_LOCAL = 32;

// ---- THE STARS ---------------------------------------------------------------------------
// THIRTY-TWO BYTES AGAIN, and for once the layout rules are not the reason. This is everything
// one copy of a mesh needs that is different from every other copy: where it is, how big it is,
// what colour it is and how brightly it burns. Sixteen floats of model matrix would say the
// same thing four times over.
struct Star {
    float x, y, z, scale;
    float r, g, b, glow;
};
static const uint32_t MAX_STARS = 40000;

struct StarPush {
    float scale, glow, pad0, pad1;
};

// SIX PLANES, AND A POINT IS INSIDE IF IT IS ON THE POSITIVE SIDE OF ALL OF THEM.
struct Plane { float a, b, c, d; };

// WHAT THE COMPUTE SHADER IS TOLD, AND WHAT THE VERTEX SHADER IS TOLD. Two different push
// blocks on two different pipelines, and neither one knows about the other.
struct DustPush {
    int32_t count, motion, shape, reset;
    float   dt, pad0, pad1, pad2;
};
struct DustDraw {
    float size, glow, pad0, pad1;
};

// ---- THE SCENES -----------------------------------------------------------------------
// Ten answers to "is this fragment in shadow", and four of them are wrong on purpose.
struct Scene {
    const char* name;
    Camera      cam;
    int         mode;          // 0 shade, 1 no shadows, 2 the shadow map itself
    uint32_t    map_size;
    float       bias;
    int         pcf;           // 0 one tap, 1 a 3x3 kernel
    bool        cull_front;    // in the SHADOW pass only
    float       sun_half;      // how much world the orthographic box covers
    float       rough_over;    // < 0 to use each part's own
    float       metal_over;
    int         curve;         // 0 clip, 1 Reinhard, 2 ACES
    float       exposure;
    float       threshold;     // luminance above which a pixel blooms
    float       bloom;         // how much of the blurred image is added back
    int         show_bloom;
    // ---- AND TODAY'S SIX, WHICH ARE ALL ABOUT A BUFFER NOTHING ON THE CPU EVER TOUCHES ----
    int         motes;         // how many, 0 for none at all
    int         motion;        // 0 drift, 1 swirl, 2 fall
    int         shape;         // 0 a box round the Meridian, 1 the beam under the lamp
    int         steps;         // how many dispatches of one step
    float       mote_size;
    float       mote_glow;
    // ---- AND TODAY'S FIVE, WHICH ARE ALL ABOUT ONE MESH DRAWN MANY TIMES ------------------
    int         stars;         // how many copies, before culling
    int         cull;          // 0 off, 1 against this camera, 2 against a FROZEN one
    float       freeze_yaw;    // how far the frozen frustum is turned from this camera
    float       star_scale;
    float       star_glow;
    const char* note;
};

// THE CAMERA IS AIMED AT THE SHADOW, not at the object. The first framing of this chapter
// looked at the Meridian from twelve units away, put the shadow in about two per cent of the
// frame, and made every comparison between two bias settings come out under one per cent --
// which the difference tool refused, correctly. A chapter about shadows points at the floor.
//
// WHERE THE SHADOW ACTUALLY IS: the globe sits at the origin, six units above a floor at
// y = -6, and the sun's direction is (-0.34, 0.52, 0.78). Following -SUN_DIR down to the floor
// takes 6 / 0.52 = 11.5 units and lands at (3.9, -6, -9.0). That is arithmetic, and it is what
// the camera is pointed at -- the second framing of this chapter, after the first one put the
// shadow near the horizon and the difference tool refused the comparisons.
// DAY 14 AIMED THIS CAMERA AT THE SHADOW, because the chapter was about the shadow. Today the
// subject is a cloud of motes filling the whole volume, so the camera pulls back and turns to
// put the Meridian in the middle of it -- and the shadow is still in frame, fifteen degrees
// below the centre, because nothing about the scene moved.
static const Camera WATCH = { 3.9f,  2.2f,  9.0f, 0.250f, -0.300f, 0.0f, 60.0f };
static const Camera CLOSE = { 3.9f, -3.4f, -3.0f, 0.000f, -0.20f, 0.0f, 68.0f };
// A CAMERA ALMOST LEVEL WITH THE FLOOR, because Fresnel is invisible head-on and overwhelming
// edge-on. A chapter about grazing angles needs a grazing angle.
static const Camera GRAZE = { 6.5f, -5.2f,  6.5f, 0.520f, -0.04f, 0.0f, 62.0f };
// A CAMERA THAT LOOKS UP THE BEAM. The cone hangs from the lamp at (0, 9.5, 0) down to the
// floor, so the one scene about it is the one scene that does not point at the floor.
static const Camera BEAM  = { 1.8f, -1.6f, 13.0f, 0.120f,  0.300f, 0.0f, 64.0f };
// A CAMERA WITH SKY IN IT. Every camera since Day 13 has pointed at the floor, because every
// chapter since Day 13 has been about something ON the floor. A shell of stars needs the
// opposite: back off, tilt up, and open the field of view.
static const Camera SKY   = { 5.0f,  1.0f, 17.0f, 0.230f,  0.115f, 0.0f, 68.0f };
// AND ONE FURTHER OUT STILL, for the two scenes that are about the SHAPE of the frustum rather
// than about what is inside it.
//
// IT IS CALLED OUTER AND NOT FAR, because windows.h defines FAR -- as nothing at all, left over
// from sixteen-bit far pointers -- and `static const Camera FAR = {...}` becomes
// `static const Camera = {...}`, which the compiler reports as a syntax error at the brace. Day
// 6 lost an afternoon to OPAQUE for exactly this reason.
static const Camera OUTER = { 16.0f, 9.0f, 34.0f, 0.380f, -0.090f, 0.0f, 72.0f };
// AND THE SAME PLACE AS SKY, TILTED FURTHER UP, because the shell is above the observatory and
// one scene should be mostly sky.
static const Camera HIGH  = { 5.0f,  1.0f, 17.0f, 0.230f,  0.420f, 0.0f, 68.0f };

static const Scene SCENES[] = {
    { "none",    SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
          0, 1, 0.0f, 1.0f, 1.0f, "no instanced draw at all" },
    { "stars",   SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 1, 0.0f, 1.0f, 1.0f, "ten thousand, one draw call" },
    { "few",     SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
       1500, 1, 0.0f, 1.6f, 1.0f, "fifteen hundred, larger" },
    { "dense",   SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      40000, 1, 0.0f, 1.0f, 1.0f, "forty thousand, still one draw call" },
    { "big",     SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 1, 0.0f, 2.6f, 1.0f, "the same ten thousand, bigger" },
    { "glow",    SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 1, 0.0f, 1.0f, 4.5f, "over 1.0, and the bloom finds them" },
    { "frozen",  SKY, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 2, 0.55f, 1.0f, 1.0f, "the frustum frozen, turned 31 degrees" },
    { "wide",    OUTER, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 1, 0.0f, 1.0f, 1.0f, "the whole shell, from outside it" },
    { "wedge",   OUTER, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 2, 0.00f, 1.0f, 1.0f, "the culled volume, seen from outside" },
    { "tilt",    HIGH, 0, 2048, 0.0022f, 1, false, 13.0f, -1.0f, -1.0f,
      2, 1.0f, 1.00f, 0.60f, 0,  4000, 0, 0,  30, 0.7f, 1.0f,
      10000, 1, 0.0f, 1.0f, 1.0f, "the same shell, looking further up" },
};
static const int SCENE_COUNT = int(sizeof SCENES / sizeof SCENES[0]);

// WHERE THE LAMPS GO. A scene is an arrangement and a count, and the buffer is filled from
// this every frame -- which is what "lights are data" means in practice.
static int fill_lamps(const Scene&, Lamp* out) {
    // THE LIGHTING STOPPED BEING THE SUBJECT, so it stops being a variable. Four warm lamps
    // and one cool one, the same in every scene, which is what lets ten pictures be about the
    // camera and nothing else.
    const float TAU = 6.28318530718f;
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        const float a = float(i) / 4.0f * TAU + 0.4f;
        out[n++] = { 6.2f * std::cos(a), 3.6f, 6.2f * std::sin(a), 30.0f,
                     1.00f, 0.93f, 0.80f, 64.0f };
    }
    out[n++] = { 0.0f, 9.5f, 0.0f, 30.0f, 0.62f, 0.78f, 1.00f, 48.0f };
    return n;
}

// THE DEPTH FORMAT. D32_SFLOAT and D16_UNORM are BOTH required to be supported as depth
// attachments on every Vulkan implementation. D24_UNORM_S8_UINT -- the one everybody reaches
// for out of habit -- is NOT: it is one of two 24-bit formats of which an implementation must
// support at least one, and choosing wrong is a crash on somebody else's card.
static const VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;

struct Rect { int32_t x0, y0, x1, y1; };
enum Fit { FIT_FILL, FIT_LETTERBOX, FIT_STRETCH };

// WHERE A SOURCE OF ONE SHAPE LANDS INSIDE A DESTINATION OF ANOTHER -- Day 3's three answers,
// kept because the window still needs them. Today every scene is the same shape as its
// destination, so this returns the whole rectangle every time.
static Rect fit_into(Fit how, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh) {
    if (how == FIT_STRETCH || how == FIT_FILL)
        return { 0, 0, int32_t(dw), int32_t(dh) };
    const double sa = double(sw) / double(sh), da = double(dw) / double(dh);
    if (da > sa) {
        const int32_t w = int32_t(double(dh) * sa + 0.5);
        return { int32_t((dw - w) / 2), 0, int32_t((dw - w) / 2) + w, int32_t(dh) };
    }
    const int32_t h = int32_t(double(dw) / sa + 0.5);
    return { 0, int32_t((dh - h) / 2), int32_t(dw), int32_t((dh - h) / 2) + h };
}

// =======================================================================================
// VALIDATION.  The most important twenty lines in the book.
// =======================================================================================
// Vulkan does almost no checking of its own: a wrong enum, a barrier you forgot, an image in
// the wrong layout -- the driver is entitled to do whatever is fastest, which is usually to
// produce a picture that is wrong, or nothing, and say nothing at all.
//
// THE VALIDATION LAYER IS A SEPARATE PIECE OF SOFTWARE that sits between your program and the
// driver and checks every call against the specification. It costs perhaps a third of the
// frame and it is worth every microsecond until Day 21. Leave it on.
static int g_messages = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    ++g_messages;
    const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR"
                                                                                  : "warn";
    // WRAPPED AT 74 COLUMNS, because a validation message is a paragraph, not a word, and the
    // half that gets cut is always the half that names the rule you broke.
    //
    // AND SPLIT ON ITS OWN NEWLINES FIRST. The message already contains line breaks -- the
    // rule, then a break, then "The Vulkan spec states:" -- and wrapping on width alone leaves
    // one of them in the middle of a wrapped line with no indent under it.
    const std::string body = d->pMessage ? d->pMessage : "";
    std::string lead = std::string("  vk ") + level + "  ";
    size_t start = 0;
    while (start <= body.size()) {
        const size_t nl = body.find('\n', start);
        std::string line = lead + body.substr(start, nl == std::string::npos
                                                     ? std::string::npos : nl - start);
        const size_t WRAP = 74;
        while (line.size() > WRAP) {
            size_t cut = line.rfind(' ', WRAP);
            if (cut == std::string::npos || cut < 12) cut = WRAP;
            std::printf("%s\n", line.substr(0, cut).c_str());
            line = "           " + line.substr(cut + 1);
        }
        std::printf("%s\n", line.c_str());
        if (nl == std::string::npos) break;
        start = nl + 1;
        lead = "           ";
    }
    return VK_FALSE;
}

// A result that is not VK_SUCCESS is a bug, and stopping on it is better than carrying on with
// a handle that was never created.
#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) {                  \
    std::printf("FAILED: %s = %d (line %d)\n", #x, (int)_r, __LINE__);               \
    std::exit(2); } } while (0)

// =======================================================================================
// THE PICTURE COMES HOME.  WIC, and the one thing it does that nobody expects.
// =======================================================================================
static bool write_png(const wchar_t* path, const uint8_t* rgba, UINT w, UINT h, UINT stride,
                      bool* swapped) {
    IWICImagingFactory* factory = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&factory)))) return false;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IWICStream* stream = nullptr;
    bool ok = false;
    if (SUCCEEDED(factory->CreateStream(&stream)) &&
        SUCCEEDED(stream->InitializeFromFilename(path, GENERIC_WRITE)) &&
        SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) &&
        SUCCEEDED(enc->Initialize(stream, WICBitmapEncoderNoCache)) &&
        SUCCEEDED(enc->CreateNewFrame(&frame, nullptr)) &&
        SUCCEEDED(frame->Initialize(nullptr)) &&
        SUCCEEDED(frame->SetSize(w, h))) {
        // ASK FOR RGBA AND READ BACK WHAT YOU WERE GIVEN. SetPixelFormat takes the format you
        // want, answers with the one you are getting, and returns S_OK either way. On this
        // machine it answers BGRA every time, so the swap is MEASURED rather than assumed.
        WICPixelFormatGUID want = GUID_WICPixelFormat32bppRGBA, got = want;
        frame->SetPixelFormat(&got);
        *swapped = (got != want);
        std::vector<uint8_t> buf(size_t(w) * h * 4);
        for (UINT y = 0; y < h; ++y)
            std::memcpy(&buf[size_t(y) * w * 4], rgba + size_t(y) * stride, size_t(w) * 4);
        if (*swapped)
            for (size_t i = 0; i < buf.size(); i += 4) std::swap(buf[i], buf[i + 2]);
        if (SUCCEEDED(frame->WritePixels(h, w * 4, (UINT)buf.size(), buf.data())) &&
            SUCCEEDED(frame->Commit()) && SUCCEEDED(enc->Commit()))
            ok = true;
    }
    if (frame) frame->Release();
    if (enc) enc->Release();
    if (stream) stream->Release();
    factory->Release();
    return ok;
}

// =======================================================================================
// MEMORY.  Vulkan has no heaps with names; it has a table, and you choose.
// =======================================================================================
// Direct3D 12 has an UPLOAD heap and a DEFAULT heap. OpenGL has usage hints. Vulkan has a list
// of memory TYPES, each with property flags and a heap it comes from, and the question is
// always the same: which types will this resource accept, and which of those has the
// properties I need? The first that satisfies both is the answer.
//
// THE ALLOWED MASK COMES FROM THE RESOURCE, NOT FROM YOU. vkGetImageMemoryRequirements fills
// in memoryTypeBits, and a type not in that mask cannot hold that image no matter how well its
// flags read.
static uint32_t memory_type(VkPhysicalDevice pd, uint32_t allowed, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties m = {};
    vkGetPhysicalDeviceMemoryProperties(pd, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
        if ((allowed & (1u << i)) && (m.memoryTypes[i].propertyFlags & want) == want)
            return i;
    std::printf("FAILED: no memory type for mask 0x%X with flags 0x%X\n", allowed, want);
    std::exit(3);
}

// A BUFFER IS A SIZE, A USAGE, AND -- SEPARATELY -- SOME MEMORY. vkCreateBuffer allocates
// nothing. It makes an object that knows how big it is and what it is for, and that object
// then TELLS YOU which memory types it will accept. Binding memory to it is a second call, and
// forgetting it is the most common first-week Vulkan bug there is.
struct Buffer {
    VkBuffer       handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
    uint32_t       type_index = ~0u;
    bool           host_visible = false;
};

// Everything the engine keeps for its whole run. Day 7 gives this a name and a lifetime; today
// it is a struct and a function that fills it.
struct Context {
    VkInstance       instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice         device = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         family = ~0u;
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkShaderModule   vs_mesh = VK_NULL_HANDLE;
    VkShaderModule   vs_shadow = VK_NULL_HANDLE;
    VkShaderModule   fs = VK_NULL_HANDLE;
    VkPipelineLayout shadow_layout = VK_NULL_HANDLE;

    // ---- THE POST CHAIN -------------------------------------------------------------------
    VkShaderModule   vs_post = VK_NULL_HANDLE;
    VkShaderModule   fs_bloom = VK_NULL_HANDLE;
    VkShaderModule   fs_tonemap = VK_NULL_HANDLE;
    VkDescriptorSetLayout post1_layout = VK_NULL_HANDLE;   // one image
    VkDescriptorSetLayout post2_layout = VK_NULL_HANDLE;   // two images
    VkPipelineLayout post1_pipe_layout = VK_NULL_HANDLE;
    VkPipelineLayout post2_pipe_layout = VK_NULL_HANDLE;
    VkSampler        post_sampler = VK_NULL_HANDLE;
    VkDescriptorPool post_pool = VK_NULL_HANDLE;
    VkPipeline       shadow_pipe = VK_NULL_HANDLE;
    VkPipeline       shadow_pipe_front = VK_NULL_HANDLE;

    // THE SHADOW MAP: one image, one view, one comparison sampler, rebuilt when a scene asks
    // for a different size.
    VkImage          shadow_img = VK_NULL_HANDLE;
    VkDeviceMemory   shadow_mem = VK_NULL_HANDLE;
    VkImageView      shadow_view = VK_NULL_HANDLE;
    VkSampler        shadow_sampler = VK_NULL_HANDLE;
    uint32_t         shadow_size = 0;
    VkDeviceSize     shadow_bytes = 0;
    VkPipelineLayout layout = VK_NULL_HANDLE;

    // TWO BUFFERS. Yesterday's nine were a tour of what a buffer can be; from today the
    // engine keeps what it draws, and it draws a cube.
    Buffer cube_vb, cube_ib;       // ONE vertex buffer and ONE index buffer for the mesh
    Buffer plane_vb, plane_ib;
    uint32_t cube_indices = 0, plane_indices = 0;
    std::vector<Part> parts;
    uint32_t mesh_vertices = 0, mesh_indices = 0;
    VkDeviceSize mesh_bytes = 0;
    uint32_t max_push = 0;

    // ---- THE TEXTURE, AND THE THREE OBJECTS IT TAKES ---------------------------------------
    VkImage        tex[MATERIAL_COUNT] = {};
    VkDeviceMemory tex_mem[MATERIAL_COUNT] = {};
    VkImageView    tex_view[MATERIAL_COUNT] = {};
    uint32_t       tex_levels = 1;
    double         mip_us = 0.0;
    VkDeviceSize   tex_bytes = 0, base_bytes = 0;
    float          max_aniso = 1.0f;
    VkSampler      sampler = VK_NULL_HANDLE;       // ONE, shared by all eight images

    // ---- TWO SETS, AND THE FREQUENCY EACH OF THEM CHANGES AT --------------------------------
    Buffer                lights_sb;                         // set 0, binding 1
    int                   lamp_count = 0;
    VkDeviceSize          sb_align = 0, max_sb = 0;
    VkDescriptorSetLayout camera_layout = VK_NULL_HANDLE;    // set 0, once a frame
    VkDescriptorSetLayout material_layout = VK_NULL_HANDLE;  // set 1, once a scene
    VkDescriptorPool      pool_desc = VK_NULL_HANDLE;
    VkDescriptorSet       camera_set = VK_NULL_HANDLE;
    VkDescriptorSet       material_set = VK_NULL_HANDLE;
    Buffer                camera_ub;
    uint32_t              limit_sets = 0, limit_images = 0;
    VkDeviceSize          ub_align = 0;
    double                bind_push_us = 0.0, bind_set_us = 0.0;
    double staged_us = 0.0;
    // THE CACHE IS AN OBJECT, not a setting. It is handed to every vkCreateGraphicsPipelines
    // call, it fills itself, and vkGetPipelineCacheData hands the bytes back for you to write
    // to disk -- which is how a shipped game skips this cost on the second launch.
    VkPipelineCache cache = VK_NULL_HANDLE;

    // ---- THE DUST: ONE BUFFER, TWO PIPELINES, AND NOTHING COPIED BETWEEN THEM --------------
    Buffer                motes;                         // STORAGE_BUFFER | VERTEX_BUFFER
    VkDescriptorSetLayout dust_layout = VK_NULL_HANDLE;  // one storage buffer, compute only
    VkDescriptorPool      dust_pool = VK_NULL_HANDLE;
    VkDescriptorSet       dust_set = VK_NULL_HANDLE;
    VkPipelineLayout      dust_comp_layout = VK_NULL_HANDLE;
    VkPipelineLayout      dust_gfx_layout = VK_NULL_HANDLE;
    VkShaderModule        cs_dust = VK_NULL_HANDLE;
    VkShaderModule        vs_dust = VK_NULL_HANDLE;
    VkShaderModule        fs_dust = VK_NULL_HANDLE;
    VkPipeline            dust_comp = VK_NULL_HANDLE;    // local_size_x 64
    VkPipeline            dust_gfx = VK_NULL_HANDLE;
    // WHAT THE CARD SAYS A DISPATCH MAY BE. Three limits and one property, and the last of
    // them -- the subgroup size -- is the one that decides what a good workgroup size is.
    uint32_t              max_invocations = 0, max_group_x = 0, max_count_x = 0, subgroup = 0;
    float                 point_lo = 0.0f, point_hi = 0.0f;
    double                gpu_step_us = 0.0, cpu_step_us = 0.0;
    double                local_us[5] = {};

    // ---- THE STARS: ONE MESH, ONE INSTANCE BUFFER, ONE DRAW --------------------------------
    Buffer           star_vb, star_ib;
    uint32_t         star_indices = 0, star_vertices = 0;
    Buffer           instances;                       // per-instance vertex binding
    VkShaderModule   vs_star = VK_NULL_HANDLE;
    VkShaderModule   fs_star = VK_NULL_HANDLE;
    VkPipelineLayout star_layout = VK_NULL_HANDLE;
    VkPipeline       star_pipe = VK_NULL_HANDLE;
    std::vector<Star> all_stars;                      // every star, before culling
    int              drawn = 0, culled = 0;
    double           cull_us = 0.0;
    double           one_draw_us = 0.0, many_draws_us = 0.0;
};

// THE SET LAYOUTS HAVE TO EXIST BEFORE THE PIPELINE LAYOUT, and the SETS cannot be written
// until the textures do. Two functions, in that order, and this declaration is what lets the
// first one be called from make_context without moving five hundred lines about.
static void make_descriptor_layouts(Context& c);

static VkShaderModule make_module(VkDevice dev, const uint32_t* words, size_t bytes) {
    VkShaderModuleCreateInfo smi = {};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = bytes;
    smi.pCode = words;
    VkShaderModule m = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &m));
    return m;
}

static void make_context(Context& c, bool validation) {
    // ---- 1. WHAT THE LOADER OFFERS ------------------------------------------------------
    uint32_t api = 0;
    VK_CHECK(vkEnumerateInstanceVersion(&api));
    std::printf("loader   : instance version %u.%u.%u\n", VK_API_VERSION_MAJOR(api),
                VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));

    uint32_t nlayers = 0;
    vkEnumerateInstanceLayerProperties(&nlayers, nullptr);
    std::vector<VkLayerProperties> layers(nlayers);
    vkEnumerateInstanceLayerProperties(&nlayers, layers.data());
    bool have = false;
    for (const VkLayerProperties& l : layers)
        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) have = true;
    std::printf("layers   : %u available, validation is %s\n", nlayers,
                have ? "PRESENT" : "ABSENT");
    validation = validation && have;

    // ---- 2. THE INSTANCE -----------------------------------------------------------------
    // THE MESSENGER IS CHAINED INTO pNext, not created afterwards. A messenger created after
    // the instance cannot report anything that went wrong DURING vkCreateInstance, which is
    // exactly when a mistyped layer name or a missing extension shows up.
    VkDebugUtilsMessengerCreateInfoEXT dbg = {};
    dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                    | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbg.pfnUserCallback = on_message;

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Meridian";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Meridian";
    // WHAT apiVersion ACTUALLY DOES: it is the highest version whose behaviour you promise to
    // understand. Ask for 1.4 and you get 1.4's rules; ask for 1.0 on a 1.4 driver and you get
    // 1.0's, including the ones that were later relaxed.
    app.apiVersion = VK_API_VERSION_1_4;

    const char* want_layers[] = { "VK_LAYER_KHRONOS_validation" };
    // PRESENTATION IS AN EXTENSION, at the instance level as well as the device level. Vulkan
    // runs on machines with no screen, so a surface is something you ask for.
    const char* want_exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                                VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_KHR_WIN32_SURFACE_EXTENSION_NAME };

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pNext = validation ? &dbg : nullptr;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = validation ? 1u : 0u;
    ici.ppEnabledLayerNames = want_layers;
    // The surface extensions are always needed; the debug one only with validation on, and
    // it is first in the array so the count can simply drop by one.
    ici.enabledExtensionCount = validation ? 3u : 2u;
    ici.ppEnabledExtensionNames = validation ? want_exts : want_exts + 1;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.instance));

    // THE MESSENGER'S OWN FUNCTION IS AN EXTENSION, so it is not in vulkan-1.lib and has to be
    // fetched. Day 2 is about why every Vulkan function works this way underneath.
    if (validation) {
        auto create = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(c.instance, "vkCreateDebugUtilsMessengerEXT");
        if (create) VK_CHECK(create(c.instance, &dbg, nullptr, &c.messenger));
    }
    std::printf("instance : created, asking for Vulkan 1.4%s\n",
                validation ? ", validation on" : ", validation OFF");

    // ---- 3. THE PHYSICAL DEVICE ----------------------------------------------------------
    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(c.instance, &ndev, nullptr);
    if (!ndev) { std::printf("FAILED: no Vulkan device\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(c.instance, &ndev, devs.data());
    std::printf("devices  : %u\n", ndev);

    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties p = {};
        vkGetPhysicalDeviceProperties(d, &p);
        const char* kind = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
                         : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
                         : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "software" : "other";
        std::printf("  %-34s %-10s api %u.%u.%u\n", p.deviceName, kind,
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion));

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i) {
            if (!(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            // A DISCRETE CARD WINS. On a laptop the integrated chip is usually first in the
            // list and will happily draw this at a fifth of the speed, which is the single
            // most common "why is it slow" in Windows graphics.
            if (c.gpu == VK_NULL_HANDLE ||
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                c.gpu = d;
                c.family = i;
            }
            break;
        }
    }
    if (c.gpu == VK_NULL_HANDLE) { std::printf("FAILED: no graphics queue\n"); std::exit(1); }

    VkPhysicalDeviceProperties gp = {};
    vkGetPhysicalDeviceProperties(c.gpu, &gp);
    std::printf("gpu      : %s\n", gp.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.gpu, &nq, qs.data());
    std::printf("queues   : %u family(ies) on this card\n", nq);
    for (uint32_t i = 0; i < nq; ++i)
        std::printf("  family %u  %2u queue(s)  %s%s%s%s\n", i, qs[i].queueCount,
                    (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "graphics " : "",
                    (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "compute " : "",
                    (qs[i].queueFlags & VK_QUEUE_TRANSFER_BIT) ? "transfer " : "",
                    (qs[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "sparse" : "");

    // ---- 4. THE LOGICAL DEVICE ------------------------------------------------------------
    // A CORE FEATURE IS STILL OFF UNTIL YOU ASK. dynamicRendering and synchronization2 have
    // been core since 1.3 and a 1.4 device still starts with them disabled. Using one without
    // enabling it is a validation error, not a crash, which is the whole argument for the
    // layer.
    VkPhysicalDeviceVulkan13Features f13 = {};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    // AND ONE MORE, FOR ONE LINE OF GLSL. `layout(local_size_x_id = 0)` compiles to an
    // OpExecutionMode called LocalSizeId, and a workgroup size that is not known until the
    // pipeline is created is only legal if maintenance4 is on. The validation layer says so
    // exactly; the driver, without the layer, says nothing and runs it anyway.
    f13.maintenance4 = VK_TRUE;

    // fillModeNonSolid is what lets a pipeline ask for lines or points instead of filled
    // triangles. It is a device feature rather than a state, and two of today's scenes need it.
    VkPhysicalDeviceFeatures base = {};
    base.fillModeNonSolid = VK_TRUE;
    // wideLines lets a pipeline set lineWidth to anything but 1.0. Like fillModeNonSolid it
    // is OPTIONAL: a card is allowed to say no, and asking for a feature the card does not
    // have makes vkCreateDevice fail rather than quietly ignore you.
    base.wideLines = VK_TRUE;
    // largePoints is what lets gl_PointSize be anything but 1.0, and like the other two it is
    // OPTIONAL. One of today's scenes draws the ring's vertices as nine-pixel points.
    base.largePoints = VK_TRUE;
    // ANISOTROPIC FILTERING IS AN OPTIONAL FEATURE. Every desktop card has had it for twenty
    // years and it is still something you ask for, because Vulkan runs on things that are not
    // desktop cards.
    base.samplerAnisotropy = VK_TRUE;

    VkPhysicalDeviceFeatures2 f2 = {};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f13;
    f2.features = base;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = c.family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    VK_CHECK(vkCreateDevice(c.gpu, &dci, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.family, 0, &c.queue);
    std::printf("device   : queue family %u, dynamic rendering and sync2 enabled\n", c.family);

    // ---- THE DISPATCH TABLE, AND EVERY CALL BELOW THIS LINE GOES STRAIGHT TO THE DRIVER ----
    // Until this call, every vkCmd* in this program was the loader's trampoline. After it, the
    // macros in vk_dispatch.h have redirected them to function pointers the driver handed over
    // -- and not one call site changed.
    const int missing = load_device_dispatch(c.device, vk);
    std::printf("dispatch : %d device-level entry point(s) from the registry, %d missing\n",
                DEVICE_DISPATCH_COUNT, missing);
    if (missing) std::exit(1);

    // ---- 5. THE PIECES EVERY SCENE SHARES --------------------------------------------------
    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = c.family;
    VK_CHECK(vkCreateCommandPool(c.device, &cpi, nullptr, &c.pool));

    make_descriptor_layouts(c);
    c.vs_mesh = make_module(c.device, kMeshVert, sizeof kMeshVert);
    c.vs_shadow = make_module(c.device, kShadowVert, sizeof kShadowVert);
    c.vs_post = make_module(c.device, kPostVert, sizeof kPostVert);
    c.fs_bloom = make_module(c.device, kBloomFrag, sizeof kBloomFrag);
    c.fs_tonemap = make_module(c.device, kTonemapFrag, sizeof kTonemapFrag);
    c.fs = make_module(c.device, kTriFrag, sizeof kTriFrag);
    c.cs_dust = make_module(c.device, kDustComp, sizeof kDustComp);
    c.vs_dust = make_module(c.device, kDustVert, sizeof kDustVert);
    c.fs_dust = make_module(c.device, kDustFrag, sizeof kDustFrag);
    c.vs_star = make_module(c.device, kStarVert, sizeof kStarVert);
    c.fs_star = make_module(c.device, kStarFrag, sizeof kStarFrag);
    std::printf("spirv    : %zu and %zu words, compiled at BUILD time\n",
                sizeof kMeshVert / 4, sizeof kTriFrag / 4);

    // ---- THE LAYOUT STOPS BEING EMPTY ------------------------------------------------------
    // A PUSH CONSTANT RANGE IS AN OFFSET, A SIZE AND WHICH STAGES CAN SEE IT. Sixty-four bytes
    // for one mat4, visible to the vertex stage. The guaranteed minimum across every Vulkan
    // implementation is 128 bytes -- two matrices -- and a range larger than the device's own
    // maxPushConstantsSize fails at layout creation, not at draw time.
    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    // THE PUSH BLOCK IS NOW PER-OBJECT DATA: a model matrix, a tint, and a material index.
    // Both stages see all of it, which is one range rather than two -- and the fragment stage
    // reads only the index. ONE MATRIX, A TINT AND TWO INTEGERS: 88 bytes, which is back
    // inside the 128 every implementation guarantees. Yesterday's 152 was the model matrix
    // sent twice, and mat3(pc.model) in the shader is the whole of the fix.
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(Mat4) + 6 * sizeof(float) + 2 * sizeof(int32_t);

    // THE DESCRIPTOR SET LAYOUTS ARE MADE ELSEWHERE and handed to the pipeline layout here.
    // A PIPELINE LAYOUT IS THE WHOLE INTERFACE: which sets, in which order, plus the push
    // ranges. Two pipelines with COMPATIBLE layouts -- identical up to the set that differs --
    // keep each other's bound sets, which is what makes set 0 worth having.
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    const VkDescriptorSetLayout layouts[2] = { c.camera_layout, c.material_layout };
    pli.setLayoutCount = 2;
    pli.pSetLayouts = layouts;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.layout));

    // THE SHADOW PASS HAS ITS OWN LAYOUT, with one push constant and no descriptor sets at
    // all. A pass that reads nothing needs nothing bound, and saying so is what makes it
    // cheap: no set is validated, no descriptor is fetched.
    VkPushConstantRange srange = {};
    srange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    srange.size = sizeof(Mat4);
    VkPipelineLayoutCreateInfo spli = {};
    spli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    spli.pushConstantRangeCount = 1;
    spli.pPushConstantRanges = &srange;
    VK_CHECK(vkCreatePipelineLayout(c.device, &spli, nullptr, &c.shadow_layout));

    VkPhysicalDeviceProperties limits = {};
    vkGetPhysicalDeviceProperties(c.gpu, &limits);
    c.max_push = limits.limits.maxPushConstantsSize;
    std::printf("push     : %u bytes of range, %u available, 128 guaranteed anywhere\n",
                uint32_t(sizeof(Mat4) + 6 * sizeof(float) + 2 * sizeof(int32_t)), c.max_push);
}

// =======================================================================================
// BUFFERS, AND THE TWO WAYS BYTES GET INTO ONE.
// =======================================================================================
static void make_buffer(Context& c, Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                        bool host_visible) {
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device, &bci, nullptr, &b.handle));

    // THE ALLOWED MASK COMES FROM THE BUFFER. memoryTypeBits is the driver saying which of the
    // card's memory types this particular buffer, with this particular usage, can live in.
    VkMemoryRequirements mr = {};
    vkGetBufferMemoryRequirements(c.device, b.handle, &mr);
    const VkMemoryPropertyFlags want =
        host_visible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // mr.size IS NOT size. The driver may want more, for alignment or for its own page
    // granularity, and allocating what you asked for rather than what it asked for is a
    // validation error on some cards and a silent corruption on others.
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits, want);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(c.device, b.handle, b.memory, 0));

    b.size = mr.size;
    b.type_index = mai.memoryTypeIndex;
    b.host_visible = host_visible;
}

static void destroy_buffer(Context& c, Buffer& b) {
    if (b.handle) vkDestroyBuffer(c.device, b.handle, nullptr);
    if (b.memory) vkFreeMemory(c.device, b.memory, nullptr);
    b.handle = VK_NULL_HANDLE;
    b.memory = VK_NULL_HANDLE;
}

// PATH ONE: MAP IT AND WRITE. Only possible on HOST_VISIBLE memory, which on a discrete card is
// across the PCI Express bus from the shader cores. Simple, immediate, and the memory the
// vertex fetch reads from is the slow one.
static void write_mapped(Context& c, Buffer& b, const void* data, size_t bytes) {
    void* dst = nullptr;
    VK_CHECK(vkMapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &dst));
    std::memcpy(dst, data, bytes);
    // HOST_COHERENT is why there is no flush here. Without it the write is not guaranteed
    // visible to the card until vkFlushMappedMemoryRanges, and leaving that out is a bug that
    // reproduces on one vendor in ten.
    vkUnmapMemory(c.device, b.memory);
}

// PATH TWO: STAGE IT. Write into a throwaway HOST_VISIBLE buffer, record a copy into the
// DEVICE_LOCAL one, submit, wait, destroy the staging buffer. Three times the code, and the
// memory the card reads from every frame afterwards is its own.
static void upload_staged(Context& c, Buffer& dst, const void* data, size_t bytes) {
    Buffer staging;
    make_buffer(c, staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    write_mapped(c, staging, data, bytes);

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    VkBufferCopy region = {};
    region.size = bytes;
    vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, fence));
    // THE WAIT IS WHY THIS BELONGS AT LOAD TIME. A staged upload in the middle of a frame that
    // waits like this has stopped the CPU dead; Day 18 does it without waiting.
    VK_CHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    destroy_buffer(c, staging);
}

// A CLOCK, because "staging is faster" is a claim and this book does not print claims.
static double now_us() {
    LARGE_INTEGER f = {}, t = {};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return double(t.QuadPart) * 1e6 / double(f.QuadPart);
}

// EVERY BUFFER THE ENGINE OWNS, BUILT ONCE. The ring vertex buffer is deliberately allocated
// larger than its contents so the wrong-stride scene reads garbage that is still INSIDE the
// buffer -- a demonstration, not a licence: a real out-of-bounds vertex fetch is undefined
// behaviour and can read another resource entirely.
static void make_geometry(Context& c, int rings, int segs, bool flat, bool loud = true) {
    MeshBuilder m;
    build_meridian(m, rings, segs);
    // FLAT SHADING COSTS THREE TIMES THE VERTICES, and the engine pays it at build time
    // rather than pretending a shader can decide.
    if (flat) flatten(m);
    c.parts = m.parts;
    c.mesh_vertices = uint32_t(m.vertices.size());
    c.mesh_indices = uint32_t(m.indices.size());
    c.mesh_bytes = VkDeviceSize(m.vertices.size() * sizeof(Vertex) + m.indices.size() * 4);

    const VkBufferUsageFlags VB = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags IB = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const double t0 = now_us();
    make_buffer(c, c.cube_vb, m.vertices.size() * sizeof(Vertex), VB, false);
    upload_staged(c, c.cube_vb, m.vertices.data(), m.vertices.size() * sizeof(Vertex));
    make_buffer(c, c.cube_ib, m.indices.size() * 4, IB, false);
    upload_staged(c, c.cube_ib, m.indices.data(), m.indices.size() * 4);
    c.staged_us = now_us() - t0;
    c.cube_indices = c.mesh_indices;

    std::vector<Vertex> pv;
    std::vector<uint32_t> pi;
    build_plane(pv, pi, 60.0f, 14.0f);
    c.plane_indices = uint32_t(pi.size());
    make_buffer(c, c.plane_vb, pv.size() * sizeof(Vertex), VB, false);
    upload_staged(c, c.plane_vb, pv.data(), pv.size() * sizeof(Vertex));
    make_buffer(c, c.plane_ib, pi.size() * 4, IB, false);
    upload_staged(c, c.plane_ib, pi.data(), pi.size() * 4);

    if (!loud) return;
    std::printf("vertex   : %zu bytes -- position, colour, uv, and a normal\n",
                sizeof(Vertex));
    std::printf("mesh     : %u vertices, %u indices, %u triangles, %zu parts\n",
                c.mesh_vertices, c.mesh_indices, c.mesh_indices / 3, c.parts.size());
    std::printf("mesh     : %llu bytes in ONE vertex buffer and ONE index buffer\n",
                (unsigned long long)c.mesh_bytes);
    std::printf("mesh     : staged in %.0f us\n", c.staged_us);
    for (const Part& pt : c.parts)
        std::printf("  %-9s first %6u  count %6u  vertexOffset %6d  material %d\n",
                    pt.name, pt.first_index, pt.index_count, pt.vertex_offset, pt.material);
}

static void destroy_geometry(Context& c) {
    destroy_buffer(c, c.cube_vb);
    destroy_buffer(c, c.cube_ib);
    destroy_buffer(c, c.plane_vb);
    destroy_buffer(c, c.plane_ib);
}

// =======================================================================================
// A TEXTURE, AND A MIP CHAIN BUILT OUT OF BLITS.
// =======================================================================================
// HOW MANY LEVELS. Halve until you reach one texel: floor(log2(max(w, h))) + 1, which for 512
// is ten. Not a choice -- a full chain is what the hardware expects and a partial one is a
// samplerLod clamp you have to remember forever.
static uint32_t mip_levels_for(uint32_t w, uint32_t h) {
    uint32_t n = 1;
    while (w > 1 || h > 1) { w = w > 1 ? w / 2 : 1; h = h > 1 ? h / 2 : 1; ++n; }
    return n;
}

static void make_texture(Context& c, int material) {
    std::vector<uint8_t> px;
    build_texture(px, material);
    c.tex_levels = mip_levels_for(TEX_SIZE, TEX_SIZE);
    c.base_bytes = VkDeviceSize(px.size());

    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { TEX_SIZE, TEX_SIZE, 1 };
    ici.mipLevels = c.tex_levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC AS WELL AS DST, because building the chain reads level n to write level
    // n+1. An image that is only a transfer destination cannot generate its own mip maps, and
    // that missing bit is the usual first failure.
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &c.tex[material]));

    VkMemoryRequirements mr = {};
    vkGetImageMemoryRequirements(c.device, c.tex[material], &mr);
    c.tex_bytes = mr.size;
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &c.tex_mem[material]));
    VK_CHECK(vkBindImageMemory(c.device, c.tex[material], c.tex_mem[material], 0));

    Buffer staging;
    make_buffer(c, staging, px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    write_mapped(c, staging, px.data(), px.size());

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    // EVERY BARRIER BELOW NAMES ONE MIP LEVEL. A layout is per subresource, not per image,
    // and that is exactly what makes generating a chain inside one command buffer possible:
    // level n is TRANSFER_SRC while level n+1 is TRANSFER_DST, in the same image, at the
    // same moment.
    auto level_barrier = [&](uint32_t level, VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags2 ss, VkAccessFlags2 sa,
                             VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = ss; b.srcAccessMask = sa;
        b.dstStageMask = ds; b.dstAccessMask = da;
        b.oldLayout = from; b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c.tex[material];
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    level_barrier(0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { TEX_SIZE, TEX_SIZE, 1 };
    vkCmdCopyBufferToImage(cmd, staging.handle, c.tex[material],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // ---- THE CHAIN: NINE BLITS, EACH HALVING THE ONE BEFORE --------------------------------
    uint32_t w = TEX_SIZE, h = TEX_SIZE;
    for (uint32_t level = 1; level < c.tex_levels; ++level) {
        level_barrier(level - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        level_barrier(level, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        const uint32_t nw = w > 1 ? w / 2 : 1, nh = h > 1 ? h / 2 : 1;
        VkImageBlit2 blit = {};
        blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 };
        blit.srcOffsets[1] = { int32_t(w), int32_t(h), 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
        blit.dstOffsets[1] = { int32_t(nw), int32_t(nh), 1 };
        VkBlitImageInfo2 bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        bi.srcImage = c.tex[material];
        bi.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bi.dstImage = c.tex[material];
        bi.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bi.regionCount = 1;
        bi.pRegions = &blit;
        // A LINEAR BLIT IS A BOX FILTER AND NOT A GOOD ONE. It is what every engine ships
        // with because it is one call; an offline tool with a Kaiser filter does better, and
        // Day 21's asset pipeline is where that would belong.
        bi.filter = VK_FILTER_LINEAR;
        vkCmdBlitImage2(cmd, &bi);

        level_barrier(level - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        w = nw; h = nh;
    }
    level_barrier(c.tex_levels - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    const double t0 = now_us();
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));
    c.mip_us = now_us() - t0;

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    destroy_buffer(c, staging);

    // THE VIEW COVERS EVERY LEVEL. levelCount = 1 compiles, runs, and gives a texture with no
    // mip maps that the sampler will happily pretend to filter -- one number, no message.
    VkImageViewCreateInfo ivi = {};
    ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image = c.tex[material];
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, c.tex_levels, 0, 1 };
    VK_CHECK(vkCreateImageView(c.device, &ivi, nullptr, &c.tex_view[material]));

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(c.gpu, &props);
    c.max_aniso = props.limits.maxSamplerAnisotropy;

}

// ---- ONE SAMPLER FOR EIGHT IMAGES ------------------------------------------------------
// Yesterday a sampler was the subject and there was one per scene. Today it is a POLICY the
// whole engine shares -- which is what a sampler is for. Most engines have fewer than ten for
// their entire lifetime.
static void make_shared_sampler(Context& c) {
    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = float(c.tex_levels);
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy = c.max_aniso;
    VK_CHECK(vkCreateSampler(c.device, &si, nullptr, &c.sampler));
}

// ---- THE SHAPES, WHICH ARE ALL THE PIPELINE LAYOUT NEEDS ------------------------------------
static void make_descriptor_layouts(Context& c) {
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(c.gpu, &props);
    c.limit_sets = props.limits.maxBoundDescriptorSets;
    c.limit_images = props.limits.maxPerStageDescriptorSampledImages;
    c.ub_align = props.limits.minUniformBufferOffsetAlignment;

    // SET 0: ONE UNIFORM BUFFER, VISIBLE TO THE VERTEX STAGE. Bound once a frame and left
    // bound, because nothing in it is about any particular object.
    // SET 0 HAS TWO BINDINGS NOW: the camera and the lights. Both change once a frame and
    // neither is about any particular object, which is what puts them in the same set.
    VkDescriptorSetLayoutBinding frame[3] = {};
    frame[0].binding = 0;
    frame[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    frame[0].descriptorCount = 1;
    frame[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    frame[1].binding = 1;
    frame[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    frame[1].descriptorCount = 1;
    frame[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // THE SHADOW MAP IS A PER-FRAME THING, so it belongs in set 0 with the camera and the
    // lights: it is rebuilt once a frame and it is not about any particular object.
    frame[2].binding = 2;
    frame[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    frame[2].descriptorCount = 1;
    frame[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dsl = {};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 3;
    dsl.pBindings = frame;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsl, nullptr, &c.camera_layout));
    dsl.bindingCount = 1;

    // SET 1: ONE BINDING HOLDING EIGHT IMAGES. descriptorCount is the array length, and the
    // shader declares sampler2D uMaterials[8] to match. Eight materials, ONE bind.
    VkDescriptorSetLayoutBinding mat = {};
    mat.binding = 0;
    mat.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    mat.descriptorCount = MATERIAL_COUNT;
    mat.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    dsl.pBindings = &mat;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsl, nullptr, &c.material_layout));

    VkDescriptorPoolSize sizes[3] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = MATERIAL_COUNT + 1;
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 2;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &c.pool_desc));

}

// ---- AND THE SETS THEMSELVES, ONCE THERE IS SOMETHING TO PUT IN THEM -------------------------
static void make_descriptors(Context& c) {
    const VkDescriptorSetLayout both[2] = { c.camera_layout, c.material_layout };
    VkDescriptorSet sets[2] = {};
    VkDescriptorSetAllocateInfo dai = {};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = c.pool_desc;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = both;
    VK_CHECK(vkAllocateDescriptorSets(c.device, &dai, sets));
    c.camera_set = sets[0];
    c.material_set = sets[1];

    // THE CAMERA BUFFER IS HOST-VISIBLE, because it changes every frame and Day 4 measured
    // what a staged upload costs when you have to wait for it.
    // 64 FOR viewProj, 64 FOR lightViewProj, 16 FOR eye, 16 FOR sun.
    make_buffer(c, c.camera_ub, 160, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    // 16 BYTES OF HEADER AND 64 LAMPS OF 32. The header is an int padded to sixteen because
    // std430 aligns the array that follows it to the size of its own members.
    make_buffer(c, c.lights_sb, 16 + MAX_LAMPS * sizeof(Lamp),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

    VkPhysicalDeviceProperties lim = {};
    vkGetPhysicalDeviceProperties(c.gpu, &lim);
    c.sb_align = lim.limits.minStorageBufferOffsetAlignment;
    c.max_sb = lim.limits.maxStorageBufferRange;

    VkDescriptorBufferInfo dbi = {};
    dbi.buffer = c.camera_ub.handle;
    dbi.offset = 0;
    dbi.range = 160;

    VkDescriptorImageInfo images[MATERIAL_COUNT] = {};
    for (int i = 0; i < MATERIAL_COUNT; ++i) {
        images[i].sampler = c.sampler;
        images[i].imageView = c.tex_view[i];
        images[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkDescriptorBufferInfo sbi = {};
    sbi.buffer = c.lights_sb.handle;
    sbi.offset = 0;
    sbi.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = c.camera_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &dbi;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = c.material_set;
    writes[1].dstBinding = 0;
    // ONE WRITE FILLS THE WHOLE ARRAY. dstArrayElement is where it starts; descriptorCount is
    // how many follow. Writing them one at a time is eight calls for no reason.
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = MATERIAL_COUNT;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = images;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = c.camera_set;
    writes[2].dstBinding = 1;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &sbi;
    vkUpdateDescriptorSets(c.device, 3, writes, 0, nullptr);

    std::printf("texture  : %d materials of %ux%u, %u level(s), %.0f us each\n",
                MATERIAL_COUNT, TEX_SIZE, TEX_SIZE, c.tex_levels, c.mip_us);
    std::printf("limits   : %u bound sets, %u sampled images a stage, UB align %llu\n",
                c.limit_sets, c.limit_images, (unsigned long long)c.ub_align);
    std::printf("sets     : set 0 is a camera UBO and a lights SSBO; set 1 is %d images\n",
                MATERIAL_COUNT);
    std::printf("storage  : %llu byte alignment, %llu bytes addressable in one binding\n",
                (unsigned long long)c.sb_align, (unsigned long long)c.max_sb);
    std::printf("lamp     : %zu bytes each in std430, %d maximum, %llu byte buffer\n",
                sizeof(Lamp), MAX_LAMPS,
                (unsigned long long)(16 + MAX_LAMPS * sizeof(Lamp)));
}

// WHAT EVERY SCENE DRAWS, WHICH TODAY IS THE SAME THING.
struct Draw { VkBuffer vb, ib; uint32_t count, first; };

static Draw draw_for(const Context& c, const Scene&) {
    return { c.cube_vb.handle, c.cube_ib.handle, c.cube_indices, 0 };
}

// ---- THE SIXTEEN FLOATS A SCENE COMES DOWN TO -----------------------------------------------
// model, then view, then projection -- and the multiplication reads RIGHT TO LEFT: the vertex
// meets the model matrix first. Writing it the other way round is the single most common
// matrix bug there is, and it produces a picture rather than an error.
// THE FROZEN FRUSTUM. Culling mode 2 tests against a camera that is NOT the one the picture is
// taken from -- the standard debug view, and the one that makes an invisible optimisation
// visible. It is the SKY camera, optionally turned, whatever camera the scene itself uses.
static Mat4 cull_view_proj(const Scene& s);

static Mat4 view_proj_for(const Scene& s) {
    const float PI = 3.14159265358979f;
    const float aspect = float(RENDER_W) / float(RENDER_H);
    const Mat4 proj = perspective(s.cam.fov_deg * PI / 180.0f, aspect, 0.1f, 200.0f, true);
    return mul(proj, camera_view(s.cam));
}

// THE SHADOW MAP'S DESCRIPTOR IS WRITTEN SEPARATELY, because the image is rebuilt whenever a
// scene asks for a different size and the set has to be told about the new one. Writing a
// descriptor set the card is not currently reading is safe; doing it mid-frame is not.
// WHICH STARS SURVIVE, AND HOW LONG IT TOOK TO FIND OUT. The survivors are copied into the
// instance buffer in order, so the draw is one contiguous run -- there is no way to tell a draw
// to skip an element, and compacting is what culling MEANS.
// DECLARED HERE, DEFINED WITH THE REST OF THE STAR MACHINERY. The cull runs at the top of
// render_scene and the planes are built two hundred lines further down; an engine is not
// obliged to be in the order it executes in, only to say so.
static void frustum_planes(const Mat4& m, Plane* out);
static bool sphere_visible(const Plane* p, float x, float y, float z, float radius);

static Mat4 cull_view_proj(const Scene& s) {
    const float PI = 3.14159265358979f;
    const float aspect = float(RENDER_W) / float(RENDER_H);
    Camera frozen = SKY;
    frozen.yaw += s.freeze_yaw;
    const Mat4 proj = perspective(frozen.fov_deg * PI / 180.0f, aspect, 0.1f, 200.0f, true);
    return mul(proj, camera_view(frozen));
}

static void cull_stars(Context& c, const Scene& s, std::vector<Star>& out) {
    out.clear();
    if (s.stars <= 0) { c.drawn = 0; c.culled = 0; c.cull_us = 0.0; return; }

    const int n = s.stars < int(MAX_STARS) ? s.stars : int(MAX_STARS);
    const double t0 = now_us();
    if (s.cull == 0) {
        out.assign(c.all_stars.begin(), c.all_stars.begin() + n);
    } else {
        Plane pl[6];
        frustum_planes(s.cull == 2 ? cull_view_proj(s) : view_proj_for(s), pl);
        out.reserve(size_t(n));
        for (int i = 0; i < n; ++i) {
            const Star& st = c.all_stars[size_t(i)];
            // THE BOUNDING RADIUS IS THE MESH'S RADIUS TIMES THE SCALES. 0.30 is what
            // gen_sphere was given; getting this too SMALL makes stars vanish at the edge of
            // the screen, which is the single most common culling bug there is.
            const float radius = 0.30f * st.scale * s.star_scale;
            if (sphere_visible(pl, st.x, st.y, st.z, radius)) out.push_back(st);
        }
    }
    c.cull_us = now_us() - t0;
    c.drawn = int(out.size());
    c.culled = n - c.drawn;
}

static void write_shadow_descriptor(Context& c) {
    VkDescriptorImageInfo dii = {};
    dii.sampler = c.shadow_sampler;
    dii.imageView = c.shadow_view;
    // THE LAYOUT IN THE DESCRIPTOR IS THE ONE THE SHADER WILL SEE, not the one the image is in
    // right now. It is in DEPTH_ATTACHMENT_OPTIMAL at this moment and will be in
    // SHADER_READ_ONLY_OPTIMAL by the time anything samples it.
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = c.camera_set;
    w.dstBinding = 2;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(c.device, 1, &w, 0, nullptr);
}

// =======================================================================================
// THE POST CHAIN: THREE PASSES THAT DRAW ONE TRIANGLE EACH.
// =======================================================================================
// EVERY ONE OF THEM READS AN IMAGE AND WRITES AN IMAGE. That is what a post-processing pass is,
// and it is why they all share two descriptor set layouts -- one image in, or two.
static void make_post(Context& c) {
    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // CLAMP_TO_EDGE, because a blur that wraps pulls the right edge of the frame into the left.
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(c.device, &si, nullptr, &c.post_sampler));

    VkDescriptorSetLayoutBinding b[2] = {};
    for (int i2 = 0; i2 < 2; ++i2) {
        b[i2].binding = uint32_t(i2);
        b[i2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i2].descriptorCount = 1;
        b[i2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl = {};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = b;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsl, nullptr, &c.post1_layout));
    dsl.bindingCount = 2;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsl, nullptr, &c.post2_layout));

    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.post1_layout;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.post1_pipe_layout));
    pli.pSetLayouts = &c.post2_layout;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.post2_pipe_layout));

    // A POOL BIG ENOUGH FOR EVERY POST SET IN A FRAME, reset between scenes rather than freed
    // one at a time -- which is Day 9's advice, taken.
    VkDescriptorPoolSize size = {};
    size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    size.descriptorCount = 64;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 32;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &c.post_pool));
}

// ONE PIPELINE FACTORY FOR ALL THREE POST PASSES. No vertex input at all, no depth, no blending,
// one triangle.
static VkPipeline make_post_pipeline(Context& c, VkShaderModule frag, VkPipelineLayout layout,
                                     VkFormat format, const VkSpecializationInfo* spec) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_post;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = spec;

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const VkDynamicState dyns[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyns;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &format;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipe));
    return pipe;
}

// =======================================================================================
// THE STARS: ONE MESH, FORTY THOUSAND COPIES, AND SIX PLANES.
// =======================================================================================
// WHERE THE STARS GO, from the same integer hash the dust uses. A shell around the observatory
// rather than a box, because a star field seen from inside a sphere looks like a sky and a
// star field seen from inside a box looks like a box.
// DECLARED HERE, DEFINED WITH THE BENCHMARK four hundred lines down. The hash that seeds the
// dust seeds the stars too, and neither of them is worth writing twice.
static float cpu_rnd(uint32_t x);

static void place_stars(std::vector<Star>& out, uint32_t n) {
    const float TAU = 6.28318530718f;
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const float th = cpu_rnd(i * 9u + 1u) * TAU;
        // cos(phi) biased upwards, so most of them are in the sky and a few are near the floor
        const float ct = cpu_rnd(i * 9u + 2u) * 1.10f - 0.18f;
        const float st = std::sqrt(1.0f - ct * ct > 0.0f ? 1.0f - ct * ct : 0.0f);
        // SQUARE ROOT ON THE RADIUS, or every star crowds the inner shell. The same correction
        // the dust made for its cone, for the same reason.
        const float rad = 16.0f + std::sqrt(cpu_rnd(i * 9u + 3u)) * 30.0f;
        const float warm = cpu_rnd(i * 9u + 5u);
        out[i].x = rad * st * std::cos(th);
        out[i].y = rad * ct + 2.0f;
        out[i].z = rad * st * std::sin(th);
        out[i].scale = 0.6f + cpu_rnd(i * 9u + 4u) * 1.1f;
        // A WIDE COLOUR SPREAD, because the tone map desaturates as it brightens and a star
        // field mixed from 0.55 to 1.0 arrives at the page as white. Day 17's Reinhard section
        // said this would happen; this is the first place in the book it had to be designed
        // around rather than described.
        out[i].r = 0.28f + 0.72f * warm;
        out[i].g = 0.42f + 0.33f * warm;
        out[i].b = 1.00f - 0.64f * warm;
        out[i].glow = 0.55f + cpu_rnd(i * 9u + 6u) * 0.85f;
    }
}

// ---- SIX PLANES OUT OF ONE MATRIX --------------------------------------------------------
// A CLIP-SPACE TEST, WRITTEN IN WORLD SPACE. After the view-projection matrix, a visible point
// satisfies -w <= x <= w, -w <= y <= w and 0 <= z <= w. Each of those six inequalities is a
// linear combination of the matrix's ROWS, and a linear combination of rows is a plane in the
// space the matrix started in. That is the whole derivation, and it is why this needs no
// trigonometry and no camera vectors.
//
// THE MATRIX IS COLUMN-MAJOR, so row i is m[0*4+i], m[1*4+i], m[2*4+i], m[3*4+i]. Getting this
// backwards gives six planes that are the TRANSPOSE's, which cull a cone-shaped region nobody
// is looking at -- and looks exactly like a culling bug with the signs wrong.
static void frustum_planes(const Mat4& m, Plane* out) {
    float r[4][4];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            r[i][k] = m.m[k * 4 + i];

    float f[6][4];
    for (int k = 0; k < 4; ++k) {
        f[0][k] = r[3][k] + r[0][k];        // left    :  x + w >= 0
        f[1][k] = r[3][k] - r[0][k];        // right   :  w - x >= 0
        f[2][k] = r[3][k] + r[1][k];        // bottom  :  y + w >= 0
        f[3][k] = r[3][k] - r[1][k];        // top     :  w - y >= 0
        f[4][k] = r[2][k];                  // near    :      z >= 0, NOT z + w
        f[5][k] = r[3][k] - r[2][k];        // far     :  w - z >= 0
    }
    // NORMALISED, so that a plane equation evaluates to a DISTANCE IN METRES rather than to
    // some multiple of one. Without this the radius test below compares a distance against a
    // radius in different units, and the culling is too eager in some directions.
    for (int i = 0; i < 6; ++i) {
        const float len = std::sqrt(f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2]);
        const float k = len > 0.0f ? 1.0f / len : 1.0f;
        out[i] = { f[i][0] * k, f[i][1] * k, f[i][2] * k, f[i][3] * k };
    }
}

// A SPHERE AGAINST SIX PLANES, and the test is the cheapest one there is: a dot product and a
// compare, six times, with an early exit the moment one of them fails.
static bool sphere_visible(const Plane* p, float x, float y, float z, float radius) {
    for (int i = 0; i < 6; ++i)
        if (p[i].a * x + p[i].b * y + p[i].c * z + p[i].d < -radius) return false;
    return true;
}

static void make_stars(Context& c) {
    // A LOW-POLY SPHERE. Eight rings and twelve segments is 192 triangles, and at ten thousand
    // copies that is 1.92 million -- which is the real budget question of an instanced draw.
    MeshBuilder m;
    m.begin();
    gen_sphere(m, 0.30f, 8, 12, 1.0f, 1.0f, 1.0f);
    m.end("star", 0);
    c.star_vertices = uint32_t(m.vertices.size());
    c.star_indices = uint32_t(m.indices.size());

    const VkBufferUsageFlags VB = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags IB = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    make_buffer(c, c.star_vb, m.vertices.size() * sizeof(Vertex), VB, false);
    upload_staged(c, c.star_vb, m.vertices.data(), m.vertices.size() * sizeof(Vertex));
    make_buffer(c, c.star_ib, m.indices.size() * 4, IB, false);
    upload_staged(c, c.star_ib, m.indices.data(), m.indices.size() * 4);

    // THE INSTANCE BUFFER IS HOST-VISIBLE, because the CPU rewrites it every frame -- the
    // survivors of the cull are a different list each time the camera moves. Day 4 measured
    // what staging costs when you have to wait for it; this is the case that cannot.
    make_buffer(c, c.instances, VkDeviceSize(MAX_STARS) * sizeof(Star),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true);

    VkPushConstantRange pr = {};
    pr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pr.size = sizeof(StarPush);
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.camera_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.star_layout));

    place_stars(c.all_stars, MAX_STARS);
}

// THE ONLY THING NEW IN THIS PIPELINE IS ONE ENUM. Two bindings instead of one, and the second
// one advances per instance.
static VkPipeline make_star_pipeline(Context& c, VkFormat format) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_star;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = c.fs_star;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binds[2] = {};
    binds[0].binding = 0;
    binds[0].stride = sizeof(Vertex);
    binds[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    binds[1].binding = 1;
    binds[1].stride = sizeof(Star);
    binds[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    // FOUR ATTRIBUTES OUT OF A SIX-FIELD VERTEX. The star shader reads the position and the
    // normal and has no use for the colour at offset 12 or the UV at offset 24, so neither is
    // described -- the stride steps over them. Describing them anyway is a validation warning.
    VkVertexInputAttributeDescription attrs[4] = {};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
    attrs[1] = { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, 32 };
    // LOCATIONS 4 AND 5 COME FROM BINDING 1. A location is a slot in the shader; a binding is
    // a buffer. Nothing requires them to line up, and here they deliberately do not.
    attrs[2] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    attrs[3] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = binds;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const VkDynamicState dyns[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyns;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // OPAQUE, AND THEY WRITE DEPTH -- unlike the dust, because a star in front of another star
    // hides it rather than adding to it.
    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &format;
    pri.depthAttachmentFormat = DEPTH_FORMAT;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = c.star_layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipe));
    return pipe;
}

// =======================================================================================
// THE DUST: A COMPUTE PIPELINE, A GRAPHICS PIPELINE, AND ONE BUFFER BETWEEN THEM.
// =======================================================================================
// A BUFFER IS BYTES. The usage flags are a list of doors it may be brought through, and a
// buffer created with STORAGE_BUFFER and VERTEX_BUFFER may be brought through both -- written
// by a compute shader as a struct array, then read by the vertex fetcher as attributes, with
// nothing copied and nothing converted. That sentence is the whole of today.
static void make_dust(Context& c) {
    make_buffer(c, c.motes, VkDeviceSize(MAX_MOTES) * sizeof(Mote),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false);

    VkDescriptorSetLayoutBinding b = {};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    // COMPUTE ONLY. The graphics side never sees this set -- it reads the same memory as
    // vertex input, which needs no descriptor at all.
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsl = {};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dsl, nullptr, &c.dust_layout));

    VkDescriptorPoolSize size = {};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dpi, nullptr, &c.dust_pool));

    VkDescriptorSetAllocateInfo dai = {};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = c.dust_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &c.dust_layout;
    VK_CHECK(vkAllocateDescriptorSets(c.device, &dai, &c.dust_set));

    VkDescriptorBufferInfo dbi = {};
    dbi.buffer = c.motes.handle;
    dbi.offset = 0;
    dbi.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = c.dust_set;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(c.device, 1, &w, 0, nullptr);

    VkPushConstantRange pr = {};
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pr.size = sizeof(DustPush);
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &c.dust_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.dust_comp_layout));

    // THE DRAWING SIDE USES SET 0 AND NOTHING ELSE -- the camera. A pipeline layout must list
    // every set the shader touches and may list no more, so this one is a single entry even
    // though the engine's main layout has two.
    VkPushConstantRange gr = {};
    gr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    gr.size = sizeof(DustDraw);
    pli.pSetLayouts = &c.camera_layout;
    pli.pPushConstantRanges = &gr;
    VK_CHECK(vkCreatePipelineLayout(c.device, &pli, nullptr, &c.dust_gfx_layout));

    // WHAT A DISPATCH IS ALLOWED TO BE, from the card rather than from a tutorial.
    VkPhysicalDeviceSubgroupProperties sg = {};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2 = {};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(c.gpu, &p2);
    c.max_invocations = p2.properties.limits.maxComputeWorkGroupInvocations;
    c.max_group_x = p2.properties.limits.maxComputeWorkGroupSize[0];
    c.max_count_x = p2.properties.limits.maxComputeWorkGroupCount[0];
    c.subgroup = sg.subgroupSize;
    c.point_lo = p2.properties.limits.pointSizeRange[0];
    c.point_hi = p2.properties.limits.pointSizeRange[1];
}

// A COMPUTE PIPELINE IS ONE STAGE AND A LAYOUT. There is no vertex input, no rasterizer, no
// blend state, no attachment format and no dynamic state -- the four hundred lines a graphics
// pipeline needs are describing fixed-function hardware this one does not use.
static VkPipeline make_dust_compute(Context& c, uint32_t local_size) {
    const VkSpecializationMapEntry entry = { 0, 0, sizeof(uint32_t) };
    VkSpecializationInfo spec = {};
    spec.mapEntryCount = 1;
    spec.pMapEntries = &entry;
    spec.dataSize = sizeof(uint32_t);
    spec.pData = &local_size;

    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = c.cs_dust;
    stage.pName = "main";
    stage.pSpecializationInfo = &spec;

    VkComputePipelineCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = stage;
    cpi.layout = c.dust_comp_layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(c.device, c.cache, 1, &cpi, nullptr, &pipe));
    return pipe;
}

// AND THE OTHER HALF: a graphics pipeline whose vertex input describes the SAME thirty-two
// bytes the compute shader writes, as two four-float attributes.
static VkPipeline make_dust_pipeline(Context& c, VkFormat format) {
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_dust;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = c.fs_dust;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bind = {};
    bind.binding = 0;
    bind.stride = sizeof(Mote);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[2] = {};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = 2;
    vi.pVertexAttributeDescriptions = attrs;

    // ONE VERTEX IS ONE PRIMITIVE. No index buffer, no winding, no culling -- a point has no
    // front and no back.
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    const VkDynamicState dyns[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dyns;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ADDITIVE, AND PREMULTIPLIED. ONE and ONE means the fragment's colour is added to what is
    // there, which is what light does and what a mote in front of another mote should do.
    VkPipelineColorBlendAttachmentState cba = {};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // TEST YES, WRITE NO -- two of Day 7's three switches, set differently from each other for
    // the first time in this book. A mote behind the Meridian must be hidden by it, and a mote
    // must not hide the mote behind IT, because they are being added together.
    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &format;
    pri.depthAttachmentFormat = DEPTH_FORMAT;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = c.dust_gfx_layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipe));
    return pipe;
}

// =======================================================================================
// THE SHADOW MAP, AND THE PASS THAT FILLS IT.
// =======================================================================================
// ONE IMAGE, USED TWO WAYS. It is a DEPTH_STENCIL_ATTACHMENT while the shadow pass writes it
// and a SAMPLED image while the main pass reads it, and the barrier between the two passes is
// what says so. That barrier is the first one in this book that is not about a layout the
// engine chose for its own convenience -- it is about two passes disagreeing.
static void make_shadow_map(Context& c, uint32_t size) {
    VkImageCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = DEPTH_FORMAT;
    ici.extent = { size, size, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &c.shadow_img));

    VkMemoryRequirements mr = {};
    vkGetImageMemoryRequirements(c.device, c.shadow_img, &mr);
    c.shadow_bytes = mr.size;
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &c.shadow_mem));
    VK_CHECK(vkBindImageMemory(c.device, c.shadow_img, c.shadow_mem, 0));

    VkImageViewCreateInfo ivi = {};
    ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivi.image = c.shadow_img;
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivi.format = DEPTH_FORMAT;
    ivi.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(c.device, &ivi, nullptr, &c.shadow_view));

    // A COMPARISON SAMPLER. compareEnable turns texture() on a sampler2DShadow into a
    // comparison rather than a fetch, and LINEAR filtering then averages the yes-or-no
    // ANSWERS rather than the depths -- which is the whole reason hardware PCF exists, because
    // the average of two depths means nothing at all.
    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // CLAMP_TO_BORDER WITH A WHITE BORDER: anything outside the map is LIT. Day 8 said this
    // mode had exactly one common use, and this is it.
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    si.compareEnable = VK_TRUE;
    si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VK_CHECK(vkCreateSampler(c.device, &si, nullptr, &c.shadow_sampler));

    c.shadow_size = size;
}

static void destroy_shadow_map(Context& c) {
    if (c.shadow_sampler) vkDestroySampler(c.device, c.shadow_sampler, nullptr);
    if (c.shadow_view) vkDestroyImageView(c.device, c.shadow_view, nullptr);
    if (c.shadow_img) vkDestroyImage(c.device, c.shadow_img, nullptr);
    if (c.shadow_mem) vkFreeMemory(c.device, c.shadow_mem, nullptr);
    c.shadow_sampler = VK_NULL_HANDLE;
    c.shadow_view = VK_NULL_HANDLE;
    c.shadow_img = VK_NULL_HANDLE;
    c.shadow_mem = VK_NULL_HANDLE;
}

// ONE VERTEX SHADER, NO FRAGMENT SHADER, NO COLOUR ATTACHMENT.
static VkPipeline make_shadow_pipeline(Context& c, bool cull_front) {
    VkPipelineShaderStageCreateInfo stage = {};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    stage.module = c.vs_shadow;
    stage.pName = "main";

    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    // ONE ATTRIBUTE. The shadow pass wants a position and nothing else, so the colour, the
    // texture coordinate and the normal are never fetched -- which is most of why a depth-only
    // pass is as cheap as it is.
    const VkVertexInputAttributeDescription attr =
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x) };

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const VkDynamicState dynamics[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynamics;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // CULLING FRONT FACES IN THE SHADOW PASS is the standard trick: the depth recorded is then
    // the BACK of each object, which is behind the surface being tested, so self-shadowing
    // acne cannot happen and no bias is needed. It only works on closed solids.
    rs.cullMode = cull_front ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    ds.maxDepthBounds = 1.0f;

    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 0;
    pri.depthAttachmentFormat = DEPTH_FORMAT;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 1;
    gpi.pStages = &stage;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = c.shadow_layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipe));
    return pipe;
}

// WHERE THE SUN LOOKS FROM. An orthographic box big enough to hold the scene, aimed down the
// sun's direction. Sizing that box is the whole of shadow-map quality: too big wastes texels
// on empty space, too small clips the shadow off at the edge.
// A LOWER SUN THROWS A LONGER SHADOW, which is both what this chapter needs and the case
// where shadow mapping is hardest: a grazing angle means more depth per texel and more bias.
static const float SUN_DIR[3] = { -0.34f, 0.52f, 0.78f };

static Mat4 sun_view_proj(float half) {
    const float dist = 26.0f;
    const float ex = SUN_DIR[0] * dist, ey = SUN_DIR[1] * dist, ez = SUN_DIR[2] * dist;
    // look_at from Day 6, which has pointed at the origin since the day it was written and
    // has never been asked to do anything else.
    const Mat4 view = look_at(ex, ey, ez);
    const Mat4 proj = ortho(half, 1.0f, 1.0f, 60.0f, true);
    return mul(proj, view);
}

// =======================================================================================
// THE PIPELINE.  Nine structs, and it cannot be changed afterwards.
// =======================================================================================
// Everything that was a call in OpenGL and a state object in Direct3D 11 is a FIELD here, and
// all of it is frozen when the pipeline is created: the shaders, the topology, the cull mode,
// the blend state, the viewport, the scissor, the format it will render into.
//
// That is not bureaucracy. It is what lets the driver compile the shaders knowing exactly what
// they will be used with, once, at load time, instead of guessing at the first draw. The price
// is that a different viewport is, strictly, a different pipeline -- which is why Day 5 is
// about dynamic state.
static VkPipeline make_pipeline(const Context& c, const Scene& s, VkFormat format) {
    // ---- THE THREE CONSTANTS THIS PIPELINE IS COMPILED WITH --------------------------------
    // A VkSpecializationInfo is a blob of bytes and a map saying which constant_id starts
    // where. The driver substitutes them into the SPIR-V and compiles it again, so a scene
    // with a different segment count is a different pipeline -- and the shader's loop bound is
    // a literal rather than a uniform read.
    // THE VERTEX STAGE HAS NO CONSTANTS ANY MORE. Everything it used to be compiled with --
    // the aspect ratio, the point size -- is now inside the matrix or gone, which is one fewer
    // reason for two scenes to need two pipelines.

    // THE FRAGMENT STAGE STILL HAS ONE. A blended scene's alpha is compiled into the
    // shader rather than passed at draw time, which is why changing it is a new pipeline.
    // TWO CONSTANTS NOW: the alpha, and the length of the material array. The second is
    // what lets the shader's modulo be a literal rather than a uniform read.
    // THE LIGHT IS A SPECIALIZATION CONSTANT, which means every scene is a different
    // compiled shader and therefore a different pipeline. That is the right choice for a book
    // -- the shader has no uniform to read and the direction is a literal in the compiled
    // code -- and the wrong one for a game, where the sun moves. Day 12 gives lights a buffer.
    // THE FALLOFF IS STILL A CONSTANT and the lamps are not. That split is the right one:
    // the falloff is a decision the engine makes once, and compiling the branch away is free;
    // the lamps move, and a moving thing belongs in a buffer.
    struct FSpec {
        float alpha; int32_t materials; int32_t falloff; float ambient;
        float bias; int32_t pcf; int32_t mode; float texel;
    };
    const FSpec fspec = { 1.0f, MATERIAL_COUNT, 3, 0.20f,
                          s.bias, s.pcf, s.mode, 1.0f / float(s.map_size) };
    const VkSpecializationMapEntry fentries[8] = {
        { 0, offsetof(FSpec, alpha),     sizeof(float)   },
        { 1, offsetof(FSpec, materials), sizeof(int32_t) },
        { 2, offsetof(FSpec, falloff),   sizeof(int32_t) },
        { 3, offsetof(FSpec, ambient),   sizeof(float)   },
        { 4, offsetof(FSpec, bias),      sizeof(float)   },
        { 5, offsetof(FSpec, pcf),       sizeof(int32_t) },
        { 6, offsetof(FSpec, mode),      sizeof(int32_t) },
        { 7, offsetof(FSpec, texel),     sizeof(float)   },
    };
    VkSpecializationInfo fsi = {};
    fsi.mapEntryCount = 8;
    fsi.pMapEntries = fentries;
    fsi.dataSize = sizeof fspec;
    fsi.pData = &fspec;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = c.vs_mesh;
    stages[0].pName = "main";           // the SPIR-V entry point, and it need not be "main"
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = c.fs;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = &fsi;

    // ---- THE VERTEX INPUT STATE, UNCHANGED IN ALL TEN SCENES -----------------------------
    // Yesterday this was the variable and the pipeline was the constant. Today it is the other
    // way round: one interleaved buffer, one binding, two attributes, and every difference
    // between the ten pictures is a field further down this function.
    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    // LOCATION 0 IS THREE FLOATS NOW. One enum, and it is the whole of "the engine is 3D".
    const VkVertexInputAttributeDescription attrs[4] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x) },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r) },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, u) },
        { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, nx) },
    };

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions = attrs;

    // THE TOPOLOGY IS FIXED TODAY AND THE POLYGON MODE IS NOT. They are easy to confuse: the
    // topology says how vertices group into primitives, the polygon mode says how a TRIANGLE
    // is filled in. Wireframe is a triangle list drawn as lines, not a line list.
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // ---- THE STATE THAT IS NOT BAKED IN -----------------------------------------------
    // VIEWPORT AND SCISSOR ARE DYNAMIC IN EVERY PIPELINE THIS ENGINE MAKES. It is what every
    // real engine does, and the reason is not speed: it is that a resized window otherwise
    // invalidates every pipeline in the program. The COUNTS are still baked in; only the
    // values move to the command buffer.
    //
    // LINE WIDTH IS DYNAMIC TOO, which makes "wireframe" and "widelines" -- identical in every
    // other field -- the only two scenes today that could share one pipeline. They do not,
    // because the engine builds one per scene, and Day 16 is where that starts to matter.
    const VkDynamicState dynamics[3] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 3;
    dyn.pDynamicStates = dynamics;

    VkPipelineViewportStateCreateInfo vps = {};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = nullptr;      // supplied by vkCmdSetViewport, not by this struct
    vps.scissorCount = 1;
    vps.pScissors = nullptr;

    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // THREE OPENGL CALLS, AS THREE FIELDS. glPolygonMode, glCullFace and glLineWidth were
    // things you did between draws; here they are things a pipeline IS.
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // CULLING IS ON FROM TODAY, and it is not a demonstration any more: the cube is a closed
    // solid, so every triangle facing away is one the camera cannot reach. Half the geometry
    // in the scene stops reaching the fragment shader, for one enum.
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;                // ignored: LINE_WIDTH is in the dynamic list

    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // ---- BLENDING IS PER ATTACHMENT, AND IT IS SIX ENUMS ---------------------------------
    // glEnable(GL_BLEND) plus glBlendFunc, as a struct: a factor for the source, a factor for
    // the destination, and an operation, separately for colour and for alpha.
    VkPipelineColorBlendAttachmentState cba = {};
    // THE OVERDRAW VIEW IS A PIPELINE, NOT A SHADER TRICK. Additive blending with the depth
    // test OFF makes the colour channel a count: every fragment that the rasterizer produced
    // adds an eighth, whether or not it would have been visible.
    const bool counting = s.mode == 1;
    cba.blendEnable = counting ? VK_TRUE : VK_FALSE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    // WITH DYNAMIC RENDERING THE PIPELINE IS TOLD A FORMAT, not a render pass. In Vulkan 1.0
    // this was a VkRenderPass object that had to exist first and had to be compatible; Day 16
    // explains what that bought and where it still matters.
    // ---- THREE FIELDS, AND PEOPLE CONFLATE ALL THREE ---------------------------------------
    // depthTestEnable  is the comparison made at all
    // depthWriteEnable does a fragment that PASSES store its own depth
    // depthCompareOp   what counts as passing
    //
    // Test on and write OFF is not a broken setting: it is exactly what a transparent object
    // wants, so that it is hidden by the wall in front of it and hides nothing itself.
    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // DAY 7 MADE THESE A SCENE PARAMETER BECAUSE THE CHAPTER WAS ABOUT THEM. Today they are
    // settings the engine has and does not vary, which is what a finished feature looks like.
    ds.depthTestEnable = counting ? VK_FALSE : VK_TRUE;
    ds.depthWriteEnable = counting ? VK_FALSE : VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    ds.minDepthBounds = 0.0f;
    ds.maxDepthBounds = 1.0f;

    VkPipelineRenderingCreateInfo pri = {};
    pri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &format;
    // THE PIPELINE IS TOLD THE DEPTH FORMAT TOO, and it has to match the attachment it will be
    // used with. A pipeline built for no depth attachment and used with one is a validation
    // error, which is the whole reason this field exists rather than being inferred.
    pri.depthAttachmentFormat = DEPTH_FORMAT;

    VkGraphicsPipelineCreateInfo gpi = {};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pColorBlendState = &cb;
    gpi.pDepthStencilState = &ds;
    gpi.pDynamicState = &dyn;
    gpi.layout = c.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    // THE SECOND PARAMETER IS THE CACHE, and passing VK_NULL_HANDLE there is what makes a cold
    // creation cold. It is not a hint: a cache that already holds this pipeline's compiled form
    // turns most of this call into a lookup.
    VK_CHECK(vkCreateGraphicsPipelines(c.device, c.cache, 1, &gpi, nullptr, &pipeline));
    return pipeline;
}

// =======================================================================================
// A WINDOW, AND THE FOUR OBJECTS BETWEEN IT AND A PICTURE.
// A COMPARISON THE ENGINE MAKES ABOUT ITS OWN OUTPUT. The chapter claims that halving the
// render size changes a certain fraction of the frame. A claim like that is worth nothing
// unless the program that made both pictures is the thing that counted, so this counts -- on
// the RGB channels, exactly, with no tolerance.
static double fraction_differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return 1.0;
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i += 4)
        if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2]) ++n;
    return double(n) / double(a.size() / 4);
}

// =======================================================================================
// =======================================================================================
static LRESULT CALLBACK window_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE || (m == WM_KEYDOWN && w == VK_ESCAPE)) { PostQuitMessage(0); return 0; }
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}

struct Window {
    HWND             hwnd = nullptr;
    VkSurfaceKHR     surface = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain = VK_NULL_HANDLE;
    VkFormat         format = VK_FORMAT_UNDEFINED;
    VkExtent2D       extent = {};
    std::vector<VkImage> images;
    VkSemaphore      acquired = VK_NULL_HANDLE;   // the image is free to write
    VkSemaphore      finished = VK_NULL_HANDLE;   // the frame is done and may be shown
};

// THE NAME OF EVERY PRESENT MODE, because the capture prints what this machine offers and a
// number is not an answer.
static const char* present_mode_name(VkPresentModeKHR m) {
    switch (m) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE   tear, never wait";
    case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX     replace the queued frame";
    case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO        a queue, and the only one";
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED  tear only if late";
    default:                               return "other";
    }
}

static bool make_window(Context& c, Window& w, uint32_t width, uint32_t height, bool show) {
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = window_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"MeridianWindow";
    RegisterClassExW(&wc);

    RECT r = { 0, 0, (LONG)width, (LONG)height };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    w.hwnd = CreateWindowExW(0, wc.lpszClassName, L"Meridian", WS_OVERLAPPEDWINDOW,
                             CW_USEDEFAULT, CW_USEDEFAULT,
                             r.right - r.left, r.bottom - r.top,
                             nullptr, nullptr, wc.hInstance, nullptr);
    if (!w.hwnd) return false;
    if (show) ShowWindow(w.hwnd, SW_SHOW);

    // ---- the surface, and the only Windows-specific Vulkan in this book --------------------
    VkWin32SurfaceCreateInfoKHR sci = {};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = wc.hInstance;
    sci.hwnd = w.hwnd;
    VK_CHECK(vkCreateWin32SurfaceKHR(c.instance, &sci, nullptr, &w.surface));

    // A QUEUE FAMILY THAT CAN DRAW IS NOT NECESSARILY ONE THAT CAN PRESENT, and the only way
    // to know is to ask about this surface. On this card family 0 can do both; on some
    // hardware the answer is two different families and every frame crosses between them.
    VkBool32 can_present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(c.gpu, c.family, w.surface, &can_present);
    std::printf("surface  : queue family %u %s present to this window\n",
                c.family, can_present ? "CAN" : "CANNOT");
    if (!can_present) return false;

    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c.gpu, w.surface, &caps);
    std::printf("surface  : %u to %u image(s), current %ux%u\n",
                caps.minImageCount, caps.maxImageCount,
                caps.currentExtent.width, caps.currentExtent.height);

    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.gpu, w.surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vkGetPhysicalDeviceSurfaceFormatsKHR(c.gpu, w.surface, &nf, formats.data());
    std::printf("surface  : %u format(s) offered\n", nf);
    w.format = formats[0].format;
    VkColorSpaceKHR space = formats[0].colorSpace;
    for (const VkSurfaceFormatKHR& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { w.format = f.format; space = f.colorSpace; }

    uint32_t nm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.gpu, w.surface, &nm, nullptr);
    std::vector<VkPresentModeKHR> modes(nm);
    vkGetPhysicalDeviceSurfacePresentModesKHR(c.gpu, w.surface, &nm, modes.data());
    std::printf("present  : %u mode(s) on this surface\n", nm);
    for (VkPresentModeKHR m : modes) std::printf("  %s\n", present_mode_name(m));

    // FIFO IS THE ONLY MODE EVERY IMPLEMENTATION MUST SUPPORT. It is a queue: one frame a
    // refresh, no tearing, and the presentation engine blocks you when it is full. Everything
    // else is a negotiation, and FIFO is the one you can rely on.
    w.extent = caps.currentExtent;
    VkSwapchainCreateInfoKHR ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = w.surface;
    ci.minImageCount = caps.minImageCount + 1 <= caps.maxImageCount || caps.maxImageCount == 0
                     ? caps.minImageCount + 1 : caps.minImageCount;
    ci.imageFormat = w.format;
    ci.imageColorSpace = space;
    ci.imageExtent = w.extent;
    ci.imageArrayLayers = 1;
    // TRANSFER_DST, NOT COLOR_ATTACHMENT. Nothing is ever drawn into a swapchain image here;
    // a finished picture is copied in. That is Day 1's decision, still holding.
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(c.device, &ci, nullptr, &w.swapchain));

    uint32_t ni = 0;
    vkGetSwapchainImagesKHR(c.device, w.swapchain, &ni, nullptr);
    w.images.resize(ni);
    vkGetSwapchainImagesKHR(c.device, w.swapchain, &ni, w.images.data());
    std::printf("swapchain: %u image(s) of %ux%u, FIFO, and you do not own them\n",
                ni, w.extent.width, w.extent.height);

    VkSemaphoreCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(c.device, &si, nullptr, &w.acquired));
    VK_CHECK(vkCreateSemaphore(c.device, &si, nullptr, &w.finished));
    return true;
}

static void destroy_window(Context& c, Window& w) {
    if (w.acquired) vkDestroySemaphore(c.device, w.acquired, nullptr);
    if (w.finished) vkDestroySemaphore(c.device, w.finished, nullptr);
    if (w.swapchain) vkDestroySwapchainKHR(c.device, w.swapchain, nullptr);
    if (w.surface) vkDestroySurfaceKHR(c.instance, w.surface, nullptr);
    if (w.hwnd) DestroyWindow(w.hwnd);
}

// ONE SCENE: DRAW INTO A SOURCE, COPY INTO A DESTINATION, READ THE DESTINATION.
// =======================================================================================
// THE SECOND IMAGE IS THE POINT. Until today the image drawn into was the image written out.
// Now there are two, and the copy between them is vkCmdBlitImage -- which is what the window
// gets as well, with a swapchain image in place of the destination.
// THE COLOUR BUFFER STOPS BEING EIGHT BITS. Sixteen-bit floats hold values well over 1.0 and
// well under it, which is what "high dynamic range" means in practice: the numbers survive until
// something decides what to do with them.
//
// R16G16B16A16_SFLOAT IS EIGHT BYTES A PIXEL rather than four, so a 1920x1080 target is 16.6 MB
// instead of 8.3. B10G11R11_UFLOAT_PACK32 is the usual compromise -- four bytes, no alpha, and
// enough range for a colour buffer -- and it is one enum away.
static bool render_scene(Context& c, const Scene& s, std::vector<uint8_t>& out, bool* swapped) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;     // what is written OUT
    const VkFormat HDR = VK_FORMAT_R16G16B16A16_SFLOAT;   // what is rendered INTO

    auto make_image = [&](uint32_t w, uint32_t h, VkImageUsageFlags usage,
                          VkImage& image, VkDeviceMemory& mem, VkImageView* view,
                          VkFormat fmt = VK_FORMAT_UNDEFINED) {
        const VkFormat use = fmt == VK_FORMAT_UNDEFINED ? FORMAT : fmt;
        const VkImageAspectFlags aspect =
            (use == VK_FORMAT_D32_SFLOAT || use == VK_FORMAT_D16_UNORM)
                ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT)
                : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = use;
        ici.extent = { w, h, 1 };
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &image));
        VkMemoryRequirements mr = {};
        vkGetImageMemoryRequirements(c.device, image, &mr);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = memory_type(c.gpu, mr.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &mem));
        VK_CHECK(vkBindImageMemory(c.device, image, mem, 0));
        if (view) {
            VkImageViewCreateInfo ivi = {};
            ivi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivi.image = image;
            ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivi.format = use;
            ivi.subresourceRange = { aspect, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(c.device, &ivi, nullptr, view));
        }
    };

    VkImage src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    VkDeviceMemory src_mem = VK_NULL_HANDLE, dst_mem = VK_NULL_HANDLE;
    VkImageView src_view = VK_NULL_HANDLE;
    // src IS NOW THE LDR OUTPUT of the tone map pass; the scene is rendered into hdr.
    make_image(RENDER_W, RENDER_H,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               src, src_mem, &src_view);

    VkImage hdr = VK_NULL_HANDLE, bloomA = VK_NULL_HANDLE, bloomB = VK_NULL_HANDLE;
    VkDeviceMemory hdr_mem = VK_NULL_HANDLE, bA_mem = VK_NULL_HANDLE, bB_mem = VK_NULL_HANDLE;
    VkImageView hdr_view = VK_NULL_HANDLE, bA_view = VK_NULL_HANDLE, bB_view = VK_NULL_HANDLE;
    const uint32_t BW = RENDER_W / 2, BH = RENDER_H / 2;
    const VkImageUsageFlags POST = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                 | VK_IMAGE_USAGE_SAMPLED_BIT;
    make_image(RENDER_W, RENDER_H, POST, hdr, hdr_mem, &hdr_view, HDR);
    // THE BLOOM BUFFERS ARE HALF RESOLUTION, which is a quarter of the pixels and is invisible
    // because the result is about to be blurred anyway. It is the cheapest four-times saving
    // in this book.
    make_image(BW, BH, POST, bloomA, bA_mem, &bA_view, HDR);
    make_image(BW, BH, POST, bloomB, bB_mem, &bB_view, HDR);
    // ---- THE SECOND ATTACHMENT -------------------------------------------------------------
    // The same size as the colour image, a different format, and it is never read back. It
    // exists only so the card can answer "is this fragment nearer than the last one".
    VkImage depth = VK_NULL_HANDLE;
    VkDeviceMemory depth_mem = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    make_image(RENDER_W, RENDER_H, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               depth, depth_mem, &depth_view, DEPTH_FORMAT);

    // THE DESTINATION IS A TRANSFER TARGET AND NOTHING IS DRAWN INTO IT. That is exactly what
    // a swapchain image is: something you copy a finished picture into.
    make_image(RENDER_W, RENDER_H,
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
               dst, dst_mem, nullptr);

    const VkDeviceSize bytes = VkDeviceSize(RENDER_W) * RENDER_H * 4;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer readback = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(c.device, &bci, nullptr, &readback));
    VkMemoryRequirements br = {};
    vkGetBufferMemoryRequirements(c.device, readback, &br);
    VkMemoryAllocateInfo bai = {};
    bai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bai.allocationSize = br.size;
    bai.memoryTypeIndex = memory_type(c.gpu, br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory readback_mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(c.device, &bai, nullptr, &readback_mem));
    VK_CHECK(vkBindBufferMemory(c.device, readback, readback_mem, 0));

    VkPipeline pipeline = make_pipeline(c, s, HDR);

    // ---- THE THREE POST PIPELINES, one per pass, each with its own constants ---------------
    struct BSpec { int32_t stage; float threshold, tx, ty; };
    const BSpec bright = { 0, s.threshold, 1.0f / float(BW), 1.0f / float(BH) };
    const BSpec blurh  = { 1, s.threshold, 1.0f / float(BW), 1.0f / float(BH) };
    const BSpec blurv  = { 2, s.threshold, 1.0f / float(BW), 1.0f / float(BH) };
    const VkSpecializationMapEntry bmap[4] = {
        { 0, offsetof(BSpec, stage),     sizeof(int32_t) },
        { 1, offsetof(BSpec, threshold), sizeof(float)   },
        { 2, offsetof(BSpec, tx),        sizeof(float)   },
        { 3, offsetof(BSpec, ty),        sizeof(float)   },
    };
    VkSpecializationInfo bsi = { 4, bmap, sizeof(BSpec), &bright };
    VkSpecializationInfo hsi = { 4, bmap, sizeof(BSpec), &blurh };
    VkSpecializationInfo vsi = { 4, bmap, sizeof(BSpec), &blurv };
    VkPipeline pipe_bright = make_post_pipeline(c, c.fs_bloom, c.post1_pipe_layout, HDR, &bsi);
    VkPipeline pipe_blurh  = make_post_pipeline(c, c.fs_bloom, c.post1_pipe_layout, HDR, &hsi);
    VkPipeline pipe_blurv  = make_post_pipeline(c, c.fs_bloom, c.post1_pipe_layout, HDR, &vsi);

    struct TSpec { int32_t curve; float exposure, bloom; int32_t show; };
    const TSpec tspec = { s.curve, s.exposure, s.bloom, s.show_bloom };
    const VkSpecializationMapEntry tmap[4] = {
        { 0, offsetof(TSpec, curve),    sizeof(int32_t) },
        { 1, offsetof(TSpec, exposure), sizeof(float)   },
        { 2, offsetof(TSpec, bloom),    sizeof(float)   },
        { 3, offsetof(TSpec, show),     sizeof(int32_t) },
    };
    VkSpecializationInfo tsi = { 4, tmap, sizeof(TSpec), &tspec };
    VkPipeline pipe_tone = make_post_pipeline(c, c.fs_tonemap, c.post2_pipe_layout,
                                              FORMAT, &tsi);

    // ---- AND THE SETS THAT POINT THEM AT EACH OTHER ----------------------------------------
    VK_CHECK(vkResetDescriptorPool(c.device, c.post_pool, 0));
    auto post_set = [&](VkDescriptorSetLayout layout, VkImageView a, VkImageView b) {
        VkDescriptorSetAllocateInfo dai = {};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = c.post_pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(c.device, &dai, &set));
        VkDescriptorImageInfo ii[2] = {};
        ii[0].sampler = c.post_sampler;
        ii[0].imageView = a;
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1] = ii[0];
        ii[1].imageView = b ? b : a;
        VkWriteDescriptorSet w[2] = {};
        for (int i2 = 0; i2 < 2; ++i2) {
            w[i2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i2].dstSet = set;
            w[i2].dstBinding = uint32_t(i2);
            w[i2].descriptorCount = 1;
            w[i2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i2].pImageInfo = &ii[i2];
        }
        vkUpdateDescriptorSets(c.device, b ? 2u : 1u, w, 0, nullptr);
        return set;
    };
    const VkDescriptorSet set_bright = post_set(c.post1_layout, hdr_view, VK_NULL_HANDLE);
    const VkDescriptorSet set_blurh  = post_set(c.post1_layout, bA_view, VK_NULL_HANDLE);
    const VkDescriptorSet set_blurv  = post_set(c.post1_layout, bB_view, VK_NULL_HANDLE);
    const VkDescriptorSet set_tone   = post_set(c.post2_layout, hdr_view, bA_view);

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    auto barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 ss, VkAccessFlags2 sa,
                       VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = ss; b.srcAccessMask = sa;
        b.dstStageMask = ds; b.dstAccessMask = da;
        b.oldLayout = from; b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    // ---- THE CULL, WHICH HAPPENS ON THE CPU AND BEFORE ANYTHING IS RECORDED ---------------
    // It is a loop over an array of thirty-two-byte structs with an early exit, which is the
    // kind of work a CPU is actually good at -- and the result is a shorter list, which is the
    // kind of thing a GPU cannot be handed any other way.
    {
        std::vector<Star> visible;
        cull_stars(c, s, visible);
        if (!visible.empty())
            write_mapped(c, c.instances, visible.data(), visible.size() * sizeof(Star));
    }

    // ---- PASS ZERO: A PASS WITH NO ATTACHMENT AND NO PIXELS ------------------------------
    // vkCmdDispatch is not inside a vkCmdBeginRendering, because there is nothing to render
    // INTO. A compute dispatch is recorded into the same command buffer as everything else and
    // is simply not part of a render pass -- which is the first time in this book that a
    // command has been legal only OUTSIDE one.
    if (s.motes > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.dust_comp);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.dust_comp_layout,
                                0, 1, &c.dust_set, 0, nullptr);

        // THE BARRIER BETWEEN TWO DISPATCHES OF THE SAME SHADER. Without it the driver is
        // free to start the second step before the first has finished writing, and a mote
        // read by one invocation while another is writing it is a race with no error message.
        auto between = [&]() {
            VkBufferMemoryBarrier2 bb = {};
            bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            bb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            bb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.buffer = c.motes.handle;
            bb.offset = 0;
            bb.size = VK_WHOLE_SIZE;
            VkDependencyInfo di = {};
            di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.bufferMemoryBarrierCount = 1;
            di.pBufferMemoryBarriers = &bb;
            vkCmdPipelineBarrier2(cmd, &di);
        };

        DustPush dp = {};
        dp.count = s.motes;
        dp.motion = s.motion;
        dp.shape = s.shape;
        dp.dt = 1.0f / 60.0f;

        // THE GROUP COUNT IS A CEILING DIVISION, and getting it wrong by rounding DOWN leaves
        // the last few motes never updated -- which looks like a handful of stuck particles
        // and is almost impossible to see in a picture of thirty thousand.
        const uint32_t groups = (uint32_t(s.motes) + DUST_LOCAL - 1) / DUST_LOCAL;

        dp.reset = 1;
        vkCmdPushConstants(cmd, c.dust_comp_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof dp, &dp);
        vkCmdDispatch(cmd, groups, 1, 1);
        between();

        dp.reset = 0;
        vkCmdPushConstants(cmd, c.dust_comp_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof dp, &dp);
        for (int step = 0; step < s.steps; ++step) {
            vkCmdDispatch(cmd, groups, 1, 1);
            between();
        }

        // AND THE ONE THAT MATTERS: from a compute shader's write to the VERTEX FETCHER's
        // read. Two different stages, two different access flags, one buffer, and no copy --
        // which is the sentence this whole chapter exists to earn.
        VkBufferMemoryBarrier2 bb = {};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bb.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        bb.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = c.motes.handle;
        bb.offset = 0;
        bb.size = VK_WHOLE_SIZE;
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &bb;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // ---- PASS ONE: DEPTH ONLY, FROM THE SUN --------------------------------------------
    // The shadow image goes from undefined to a depth attachment, gets written, and then
    // becomes a texture the fragment shader can read. Two barriers and two layouts for one
    // image inside one command buffer, which is what a two-pass frame IS.
    {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c.shadow_img;
        b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    {
        VkRenderingAttachmentInfo sd = {};
        sd.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        sd.imageView = c.shadow_view;
        sd.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        sd.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // STORE, not DONT_CARE: this one IS read afterwards, which is the only depth
        // attachment in this book that is.
        sd.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        sd.clearValue.depthStencil.depth = 1.0f;

        VkRenderingInfo sri = {};
        sri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        sri.renderArea = { { 0, 0 }, { c.shadow_size, c.shadow_size } };
        sri.layerCount = 1;
        sri.colorAttachmentCount = 0;
        sri.pDepthAttachment = &sd;

        vkCmdBeginRendering(cmd, &sri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          s.cull_front ? c.shadow_pipe_front : c.shadow_pipe);
        VkViewport svp = { 0.0f, 0.0f, float(c.shadow_size), float(c.shadow_size), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &svp);
        VkRect2D ssr = { { 0, 0 }, { c.shadow_size, c.shadow_size } };
        vkCmdSetScissor(cmd, 0, 1, &ssr);

        const Mat4 lvp = sun_view_proj(s.sun_half);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &c.cube_vb.handle, &zero);
        vkCmdBindIndexBuffer(cmd, c.cube_ib.handle, 0, VK_INDEX_TYPE_UINT32);
        for (const Part& pt : c.parts) {
            const Mat4 m = mul(lvp, mul(translate(pt.ox, pt.oy, pt.oz),
                                        rotate_x(pt.oy != 0.0f ? 0.0f : 0.18f)));
            vkCmdPushConstants(cmd, c.shadow_layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof m, &m);
            vkCmdDrawIndexed(cmd, pt.index_count, 1, pt.first_index, pt.vertex_offset, 0);
        }
        // THE FLOOR IS NOT IN THE SHADOW PASS. It cannot shadow anything -- everything in this
        // scene is above it -- and leaving it out is one fewer draw and one fewer surface for
        // the bias to be wrong on.
        vkCmdEndRendering(cmd);
    }

    {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c.shadow_img;
        b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    barrier(hdr, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    // THE DEPTH IMAGE NEEDS ITS OWN BARRIER, ITS OWN LAYOUT AND ITS OWN ASPECT. Reusing the
    // colour helper here with VK_IMAGE_ASPECT_COLOR_BIT is a validation error that reads like
    // a typo and behaves like a corrupted frame.
    {
        VkImageMemoryBarrier2 b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depth;
        b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    VkRenderingAttachmentInfo att = {};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = hdr_view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    std::memcpy(att.clearValue.color.float32, NIGHT, sizeof NIGHT);

    // THE CLEAR VALUE HAS TO AGREE WITH THE COMPARISON. Cleared to 1.0 with compareOp LESS,
    // the first fragment at any pixel passes. Cleared to 1.0 with GREATER, NOTHING passes --
    // a black frame, no validation message, and an afternoon gone. The GREATER scene below
    // clears to 0.0 instead, which is exactly the pairing reverse-Z depth buffers use.
    VkRenderingAttachmentInfo datt = {};
    datt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    datt.imageView = depth_view;
    datt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    datt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // STORE_OP_DONT_CARE, because nothing reads this image after the pass. On a tiled GPU
    // that is the difference between writing 8 MB to memory and not writing it.
    datt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    datt.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { { 0, 0 }, { RENDER_W, RENDER_H } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    // THE ATTACHMENT IS ALWAYS THERE, even for the scenes that do not test against it. A
    // pipeline that declares a depth format must be used inside a rendering that has one, and
    // "turn depth off" means turning the TEST off, not removing the image.
    ri.pDepthAttachment = &datt;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // ---- BINDING THE BYTES, AND THEN ASKING FOR THEM ---------------------------------------
    // vkCmdBindVertexBuffers takes an ARRAY of buffers and an array of offsets, because a
    // pipeline can have several bindings. The offset is why one big buffer can hold every mesh
    // in a scene and each draw simply starts further in -- which is what Day 10 does.
    const Draw d = draw_for(c, s);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &d.vb, &off);
    vkCmdBindIndexBuffer(cmd, d.ib, 0, VK_INDEX_TYPE_UINT32);
    // A PLANE HAS ONE SIDE THAT MATTERS AND CULLING WOULD HIDE HALF THE SCENES, so today's
    // pipelines do not cull. Day 10 turns it back on with a mesh that has an inside.

    // ---- SET 0 AND SET 1, BOUND ONCE, AND THEN SIXTY-FOUR PUSHES --------------------------
    // The camera goes into its uniform buffer here rather than into the push block, because
    // every object shares it and re-sending a matrix sixty-four times is sixty-three copies
    // nobody asked for.
    struct CameraUB { Mat4 view_proj; Mat4 light_view_proj; float eye[4]; float sun[4]; };
    CameraUB cam = {};
    cam.view_proj = view_proj_for(s);
    cam.light_view_proj = sun_view_proj(s.sun_half);
    cam.eye[0] = s.cam.x;
    cam.eye[1] = s.cam.y;
    cam.eye[2] = s.cam.z;
    cam.sun[0] = SUN_DIR[0];
    cam.sun[1] = SUN_DIR[1];
    cam.sun[2] = SUN_DIR[2];
    cam.sun[3] = 3.40f;
    write_mapped(c, c.camera_ub, &cam, sizeof cam);

    // ---- THE LAMPS, WRITTEN EVERY FRAME ---------------------------------------------------
    // Sixteen bytes of header and up to sixty-four lamps. It is a memcpy into host-visible
    // memory, and it is the whole of "the lights moved" -- no pipeline, no descriptor, no
    // barrier, because nothing on the card is reading it while this runs.
    struct LightsSB { int32_t count, pad0, pad1, pad2; Lamp lamps[MAX_LAMPS]; } lb = {};
    lb.count = fill_lamps(s, lb.lamps);
    write_mapped(c, c.lights_sb, &lb, sizeof(int32_t) * 4 + sizeof(Lamp) * size_t(lb.count));
    c.lamp_count = lb.count;

    VkViewport vp = { 0.0f, 0.0f, float(RENDER_W), float(RENDER_H), 0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sr = { { 0, 0 }, { RENDER_W, RENDER_H } };
    vkCmdSetScissor(cmd, 0, 1, &sr);
    vkCmdSetLineWidth(cmd, 1.0f);

    // TWO SETS IN ONE CALL, starting at set 0. Binding them separately is also legal and is
    // what an engine does when they change at different moments -- which is the whole reason
    // they are separate sets.
    const VkDescriptorSet sets[2] = { c.camera_set, c.material_set };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c.layout, 0, 2,
                            sets, 0, nullptr);

    // A NORMAL MATRIX IS NOT THE MODEL MATRIX, except when it is. For a rotation and a
    // uniform scale -- which is everything in this book -- the model matrix's own 3x3 rotates
    // a normal correctly. For a NON-UNIFORM scale it does not: squash a sphere along y and its
    // normals must be squashed along the OTHER axes, which is the inverse transpose.
    //
    // This engine sends the model matrix again under a second name, and says so, rather than
    // sending an inverse transpose it does not need and cannot demonstrate.
    struct Push {
        Mat4 model;
        float tint[4];
        int32_t material;
        int32_t mode;
        float roughness;
        float metalness;
    };

    // THE FLOOR FIRST, and it is the one thing today that is not part of the mesh.
    {
        const Draw pd = { c.plane_vb.handle, c.plane_ib.handle, c.plane_indices, 0 };
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &pd.vb, &zero);
        vkCmdBindIndexBuffer(cmd, pd.ib, 0, VK_INDEX_TYPE_UINT32);
        Push push = {};
        push.model = translate(0.0f, -6.0f, 0.0f);
        push.tint[0] = push.tint[1] = push.tint[2] = 0.45f;
        push.tint[3] = 1.0f;
        push.material = 3;
        push.mode = 0;
        // THE FLOOR IS THE ONE SURFACE THAT SHOWS FRESNEL ACROSS THE WHOLE FRAME, because it
        // runs to the horizon and the angle to it changes all the way there.
        push.roughness = s.rough_over >= 0.0f ? s.rough_over : 0.40f;
        push.metalness = s.metal_over >= 0.0f ? s.metal_over : 0.0f;
        vkCmdPushConstants(cmd, c.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof push, &push);
        vkCmdDrawIndexed(cmd, pd.count, 1, pd.first, 0, 0);
    }

    vkCmdBindVertexBuffers(cmd, 0, 1, &d.vb, &off);
    vkCmdBindIndexBuffer(cmd, d.ib, 0, VK_INDEX_TYPE_UINT32);

    // ---- NINE PARTS, NINE DRAWS, AND NOT ONE REBIND BETWEEN THEM -------------------------
    // The vertex buffer and the index buffer were bound once, above. Everything below is
    // firstIndex, indexCount and vertexOffset -- three integers a part, which is exactly what
    // Day 4 said an engine keeps.
    for (int copy = 0; copy < 1; ++copy) {
        const float ring = 6.4f;
        const float a = 1 == 1 ? 0.0f
                                      : float(copy) / float(1) * 6.2831853f;
        const float cx = 1 == 1 ? 0.0f : ring * std::cos(a);
        const float cz = 1 == 1 ? 0.0f : ring * std::sin(a);

        for (size_t pi = 0; pi < c.parts.size(); ++pi) {
            const Part& pt = c.parts[pi];
            // only >= 0 draws that one part; -2 is the three rings; -3 is the stand.
            if ((-1) >= 0 && int(pi) != (-1)) continue;
            if ((-1) == -2 && (pi < 1 || pi > 3)) continue;
            if ((-1) == -3 && pi < 6) continue;

            // EXPLODING IS A TRANSLATION ALONG THE PART'S OWN OFFSET, which is why a Part
            // carries one. A mesh with no idea where its pieces belong cannot be taken apart.
            const float k = 1.0f + 0.0f;
            Push push = {};
            // THE OBJECT NO LONGER TURNS WITH THE SCENE. For twelve days the scene table
            // rotated the world to show it from another angle; today the CAMERA moves, which
            // is what a camera is for.
            push.model = mul(translate(cx + pt.ox * k, pt.oy * k, cz + pt.oz * k),
                             rotate_x(pt.oy != 0.0f ? 0.0f : 0.18f));
                push.tint[0] = push.tint[1] = push.tint[2] = 1.0f;
            push.tint[3] = 1.0f;
            push.material = pt.material;
            push.mode = 0;
            push.roughness = s.rough_over >= 0.0f ? s.rough_over : pt.roughness;
            push.metalness = s.metal_over >= 0.0f ? s.metal_over : pt.metalness;
            vkCmdPushConstants(cmd, c.layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof push, &push);
            // THE FOURTH PARAMETER IS vertexOffset, and it is added to every index in the
            // slice BEFORE the fetch. Without it each generator would have to know where in
            // the shared buffer it had landed.
            vkCmdDrawIndexed(cmd, pt.index_count, 1, pt.first_index, pt.vertex_offset, 0);
        }
    }

    // ---- AND THE DUST, LAST, BECAUSE IT IS ADDITIVE AND DOES NOT WRITE DEPTH ---------------
    // One draw call for every mote in the scene. Nothing about this call knows the positions
    // came from a compute shader -- it binds a vertex buffer and asks for N vertices, exactly
    // as Day 4 did.
    if (s.motes > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c.dust_gfx);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c.dust_gfx_layout,
                                0, 1, &c.camera_set, 0, nullptr);
        DustDraw dd = {};
        dd.size = s.mote_size;
        dd.glow = s.mote_glow;
        vkCmdPushConstants(cmd, c.dust_gfx_layout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof dd, &dd);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &c.motes.handle, &zero);
        vkCmdDraw(cmd, uint32_t(s.motes), 1, 0, 0);
    }

    // ---- AND TEN THOUSAND STARS, IN ONE CALL ----------------------------------------------
    // TWO BUFFERS BOUND AT ONCE, which vkCmdBindVertexBuffers has taken an array for since
    // Day 4. Binding 0 advances per vertex, binding 1 per instance, and the pipeline is what
    // knows the difference.
    if (c.drawn > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c.star_pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c.star_layout,
                                0, 1, &c.camera_set, 0, nullptr);
        StarPush sp = {};
        sp.scale = s.star_scale;
        sp.glow = s.star_glow;
        vkCmdPushConstants(cmd, c.star_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof sp, &sp);
        const VkBuffer vbs[2] = { c.star_vb.handle, c.instances.handle };
        const VkDeviceSize offs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, offs);
        vkCmdBindIndexBuffer(cmd, c.star_ib.handle, 0, VK_INDEX_TYPE_UINT32);
        // THE SECOND PARAMETER. It has been 1 since Day 5 and nobody looked at it.
        vkCmdDrawIndexed(cmd, c.star_indices, uint32_t(c.drawn), 0, 0, 0);
    }
    vkCmdEndRendering(cmd);

    // ---- PASSES 3a, 3b, 3c AND 4 -----------------------------------------------------------
    // Each one is: make the source readable, make the target writable, draw one triangle.
    // The barriers are identical in shape and differ only in which image they name, which is
    // exactly the repetition a frame graph exists to remove.
    auto to_read = [&](VkImage im) {
        barrier(im, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    };
    auto to_write = [&](VkImage im) {
        barrier(im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    };
    auto full_screen = [&](VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                           VkImageView target, uint32_t w, uint32_t h) {
        VkRenderingAttachmentInfo a = {};
        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView = target;
        a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // every pixel is written
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri2 = {};
        ri2.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri2.renderArea = { { 0, 0 }, { w, h } };
        ri2.layerCount = 1;
        ri2.colorAttachmentCount = 1;
        ri2.pColorAttachments = &a;
        vkCmdBeginRendering(cmd, &ri2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        VkViewport vp2 = { 0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp2);
        VkRect2D sr2 = { { 0, 0 }, { w, h } };
        vkCmdSetScissor(cmd, 0, 1, &sr2);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                                &set, 0, nullptr);
        // THREE VERTICES AND NO BUFFER.
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    };

    to_read(hdr);
    to_write(bloomA);
    full_screen(pipe_bright, c.post1_pipe_layout, set_bright, bA_view, BW, BH);
    to_read(bloomA);
    to_write(bloomB);
    full_screen(pipe_blurh, c.post1_pipe_layout, set_blurh, bB_view, BW, BH);
    to_read(bloomB);
    to_write(bloomA);
    full_screen(pipe_blurv, c.post1_pipe_layout, set_blurv, bA_view, BW, BH);
    to_read(bloomA);
    to_write(src);
    full_screen(pipe_tone, c.post2_pipe_layout, set_tone, src_view, RENDER_W, RENDER_H);

    barrier(src, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    barrier(dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // THE BARS IN A LETTERBOX ARE NOT DRAWN. They are whatever the destination already held,
    // so it is cleared first and the copy lands in the middle of it. A swapchain image is the
    // same: what you do not write is what the last frame left.
    VkClearColorValue bar = {};
    bar.float32[0] = 0.02f; bar.float32[1] = 0.02f; bar.float32[2] = 0.03f;
    bar.float32[3] = 1.0f;
    VkImageSubresourceRange whole = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bar, 1, &whole);

    const Rect r = fit_into(FIT_FILL, RENDER_W, RENDER_H, RENDER_W, RENDER_H);
    // A BLIT NAMES TWO RECTANGLES AND NOTHING REQUIRES THEM TO MATCH. Swapping the
    // destination's y values is how a copy is flipped, and forgetting that you have already
    // flipped once is how a picture ends up upside down.
    VkImageBlit2 region = {};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
    region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.srcOffsets[0] = { 0, 0, 0 };
    region.srcOffsets[1] = { int32_t(RENDER_W), int32_t(RENDER_H), 1 };
    region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.dstOffsets[0] = { r.x0, r.y0, 0 };
    region.dstOffsets[1] = { r.x1, r.y1, 1 };

    VkBlitImageInfo2 blit = {};
    blit.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
    blit.srcImage = src;
    blit.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blit.dstImage = dst;
    blit.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blit.regionCount = 1;
    blit.pRegions = &region;
    blit.filter = VK_FILTER_LINEAR;
    vkCmdBlitImage2(cmd, &blit);

    barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region2 = {};
    region2.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region2.imageExtent = { RENDER_W, RENDER_H, 1 };
    vkCmdCopyImageToBuffer(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region2);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, UINT64_MAX));

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(c.device, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    out.assign((const uint8_t*)mapped, (const uint8_t*)mapped + bytes);
    wchar_t path[256];
    swprintf(path, 256, L"frame_%S.png", s.name);
    const bool ok = write_png(path, out.data(), RENDER_W, RENDER_H, RENDER_W * 4, swapped);
    vkUnmapMemory(c.device, readback_mem);

    vkDestroyFence(c.device, fence, nullptr);
    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    vkDestroyPipeline(c.device, pipeline, nullptr);
    vkDestroyBuffer(c.device, readback, nullptr);
    vkFreeMemory(c.device, readback_mem, nullptr);
    vkDestroyPipeline(c.device, pipe_bright, nullptr);
    vkDestroyPipeline(c.device, pipe_blurh, nullptr);
    vkDestroyPipeline(c.device, pipe_blurv, nullptr);
    vkDestroyPipeline(c.device, pipe_tone, nullptr);
    vkDestroyImageView(c.device, hdr_view, nullptr);
    vkDestroyImageView(c.device, bA_view, nullptr);
    vkDestroyImageView(c.device, bB_view, nullptr);
    vkDestroyImage(c.device, hdr, nullptr);
    vkDestroyImage(c.device, bloomA, nullptr);
    vkDestroyImage(c.device, bloomB, nullptr);
    vkFreeMemory(c.device, hdr_mem, nullptr);
    vkFreeMemory(c.device, bA_mem, nullptr);
    vkFreeMemory(c.device, bB_mem, nullptr);
    vkDestroyImageView(c.device, depth_view, nullptr);
    vkDestroyImage(c.device, depth, nullptr);
    vkFreeMemory(c.device, depth_mem, nullptr);
    vkDestroyImageView(c.device, src_view, nullptr);
    vkDestroyImage(c.device, src, nullptr);
    vkFreeMemory(c.device, src_mem, nullptr);
    vkDestroyImage(c.device, dst, nullptr);
    vkFreeMemory(c.device, dst_mem, nullptr);
    return ok;
}

// =======================================================================================
// WHAT A DISPATCH COSTS, AND WHETHER IT IS DOING THE SAME ARITHMETIC.
// =======================================================================================
// THE SAME HASH AGAIN, IN C++. Two copies of one function in two languages, and keeping them
// in agreement is the price of being able to check one against the other. Everything about
// them is exact integer arithmetic, which is why the seeding can be compared to the bit.
static uint32_t cpu_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
static float cpu_rnd(uint32_t x) { return float(cpu_hash(x) & 0xffffffu) / 16777216.0f; }

static void cpu_seed(std::vector<Mote>& m, uint32_t n) {
    m.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const float a = cpu_rnd(i * 3u + 0u);
        const float b = cpu_rnd(i * 3u + 1u);
        const float d = cpu_rnd(i * 3u + 2u);
        m[i].x = (a - 0.5f) * 27.0f;
        m[i].y = -5.6f + b * 14.0f;
        m[i].z = (d - 0.5f) * 27.0f;
        m[i].size = 0.55f + cpu_rnd(i * 5u + 1u) * 0.85f;
        const float warm = cpu_rnd(i * 3u + 7u);
        m[i].r = 0.62f + (1.00f - 0.62f) * warm;
        m[i].g = 0.74f + (0.86f - 0.74f) * warm;
        m[i].b = 1.00f + (0.62f - 1.00f) * warm;
        m[i].life = 0.35f + cpu_rnd(i * 5u + 3u) * 0.65f;
    }
}

static void cpu_step(std::vector<Mote>& m, uint32_t n, float dt) {
    for (uint32_t i = 0; i < n; ++i) {
        Mote& q = m[i];
        const float vx = 0.55f * std::sin(q.y * 0.65f + float(i & 63u));
        const float vz = 0.55f * std::cos(q.x * 0.55f + float(i & 31u));
        q.x += vx * dt;
        q.y += 0.85f * dt;
        q.z += vz * dt;
        if (q.y > 9.6f) q.y = -5.9f;
        if (q.y < -5.94f) q.y = -5.94f;
    }
}

// ONE COMMAND BUFFER, N DISPATCHES, ONE SUBMIT AND ONE WAIT. The queue is timed from the CPU,
// so the number includes the submit and the wait -- which at two hundred dispatches is noise
// and at one dispatch would be the whole measurement. Day 2's lesson, again.
static double time_dispatches(Context& c, VkPipeline pipe, uint32_t motes, uint32_t local,
                              int reps) {
    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = c.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cmd));

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.dust_comp_layout,
                            0, 1, &c.dust_set, 0, nullptr);
    DustPush dp = {};
    dp.count = int32_t(motes);
    dp.dt = 1.0f / 60.0f;
    vkCmdPushConstants(cmd, c.dust_comp_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof dp, &dp);
    const uint32_t groups = (motes + local - 1) / local;
    for (int i = 0; i < reps; ++i) {
        vkCmdDispatch(cmd, groups, 1, 1);
        VkBufferMemoryBarrier2 bb = {};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                         | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = c.motes.handle;
        bb.offset = 0;
        bb.size = VK_WHOLE_SIZE;
        VkDependencyInfo di = {};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.bufferMemoryBarrierCount = 1;
        di.pBufferMemoryBarriers = &bb;
        vkCmdPipelineBarrier2(cmd, &di);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo csi = {};
    csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;

    const double t = now_us();
    VK_CHECK(vkQueueSubmit2(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    const double us = now_us() - t;

    vkFreeCommandBuffers(c.device, c.pool, 1, &cmd);
    return us / double(reps);
}

static void measure_dispatch(Context& c) {
    const uint32_t N = 150000;
    const int REPS = 200;

    std::printf("compute  : max %u invocation(s) a group, max %u in x, %u group(s) in x\n",
                c.max_invocations, c.max_group_x, c.max_count_x);
    std::printf("compute  : subgroup %u -- a workgroup not a multiple of it wastes lanes\n",
                c.subgroup);
    std::printf("points   : gl_PointSize may be %.0f to %.0f pixels on this card\n",
                c.point_lo, c.point_hi);
    std::printf("motes    : %u of %u bytes -- %.2f MB, device local, never mapped\n",
                MAX_MOTES, unsigned(sizeof(Mote)),
                double(MAX_MOTES) * sizeof(Mote) / 1048576.0);

    // ---- FIVE WORKGROUP SIZES, AND THE SAME WORK EACH TIME --------------------------------
    static const uint32_t SIZES[5] = { 1, 8, 32, 64, 256 };
    for (int i = 0; i < 5; ++i) {
        VkPipeline pipe = make_dust_compute(c, SIZES[i]);
        time_dispatches(c, pipe, N, SIZES[i], 8);          // warm the card up first
        c.local_us[i] = time_dispatches(c, pipe, N, SIZES[i], REPS);
        vkDestroyPipeline(c.device, pipe, nullptr);
        std::printf("local    : %3u -> %7.1f us a step  (%6u group(s))\n",
                    SIZES[i], c.local_us[i], (N + SIZES[i] - 1) / SIZES[i]);
    }
    c.gpu_step_us = c.local_us[3];

    // ---- AND THE SAME ARITHMETIC ON THE CPU -----------------------------------------------
    std::vector<Mote> host;
    cpu_seed(host, N);
    const double t = now_us();
    for (int i = 0; i < 20; ++i) cpu_step(host, N, 1.0f / 60.0f);
    c.cpu_step_us = (now_us() - t) / 20.0;

    std::printf("step     : GPU %7.1f us   CPU %7.1f us   %.0fx\n",
                c.gpu_step_us, c.cpu_step_us, c.cpu_step_us / c.gpu_step_us);
    std::printf("step     : %u motes -- %.1f million a second on the card\n",
                N, double(N) / c.gpu_step_us);
}

// =======================================================================================
// WHAT A PIPELINE COSTS, AND WHAT THE CACHE TAKES IT DOWN TO.
// =======================================================================================
// Three passes over the same ten scenes. The first has no cache at all, which is the price a
// program pays on a machine it has never run on. The second fills a cache. The third uses it.
//
// THE CACHE IS NOT A SPEED SETTING. It is a blob of driver-specific bytes you are expected to
// write to disk and hand back next launch -- which is what "shader compilation" progress bars
// in shipped games are doing, and why they only appear once.
static void measure_pipelines(Context& c) {
    const VkFormat FORMAT = VK_FORMAT_R8G8B8A8_UNORM;

    c.cache = VK_NULL_HANDLE;
    double t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double cold = now_us() - t;

    VkPipelineCacheCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK(vkCreatePipelineCache(c.device, &pci, nullptr, &c.cache));

    t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double filling = now_us() - t;

    t = now_us();
    for (const Scene& s : SCENES)
        vkDestroyPipeline(c.device, make_pipeline(c, s, FORMAT), nullptr);
    const double hot = now_us() - t;

    size_t bytes = 0;
    vkGetPipelineCacheData(c.device, c.cache, &bytes, nullptr);

    std::printf("pipeline : %d scene(s), each one a separate compiled object\n", SCENE_COUNT);
    std::printf("pipeline : cold %7.0f us   filling %7.0f us   cached %7.0f us\n",
                cold, filling, hot);
    std::printf("pipeline : %.0f us each cold, %.0f us each cached\n",
                cold / SCENE_COUNT, hot / SCENE_COUNT);
    std::printf("cache    : %zu bytes, and this is what a game writes to disk\n", bytes);
}

// =======================================================================================
// WHAT A BIND COSTS, AGAINST WHAT A PUSH COSTS.
// =======================================================================================
// THE CLAIM THIS CHAPTER MAKES is that an array in set 1 is worth having because it turns a
// descriptor-set bind per object into a push constant per object. That is a claim about a
// cost, so it gets measured.
//
// AND IT GETS ITS OWN INSTANCE AND DEVICE, WITH NO LAYERS. The first version of this ran on
// the engine's own device and reported 2,367 nanoseconds a bind -- which is the validation
// layer intercepting every call, not the driver doing anything. Day 2 made the identical
// mistake with the loader's trampoline and this is the identical fix.
//
// THE SECOND DEVICE HAS NO TEXTURES, so both of its sets hold uniform buffers. What is being
// measured is what vkCmdBindDescriptorSets costs to RECORD, and that does not depend on what
// the set points at.
// EVERY CALL BELOW IS WRITTEN (::vkName)(...) WITH THE PARENTHESES. `::vkName(` does NOT stop
// the preprocessor: the dispatch macros are function-like, so ::vkResetCommandBuffer(cmd, 0)
// expands to ::vk.vkResetCommandBuffer(cmd, 0) and calls the ENGINE's device function on a
// command buffer belonging to a different device. The first version of this crashed with
// 0xC0000409 and no message at all.
static void measure_binding(Context& c) {
    const int DRAWS = 4096;

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Meridian bench";
    app.apiVersion = VK_API_VERSION_1_4;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VK_CHECK((::vkCreateInstance)(&ici, nullptr, &inst));

    uint32_t n = 0;
    (::vkEnumeratePhysicalDevices)(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    (::vkEnumeratePhysicalDevices)(inst, &n, devs.data());
    VkPhysicalDevice gpu = devs[0];
    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties pr = {};
        (::vkGetPhysicalDeviceProperties)(d, &pr);
        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { gpu = d; break; }
    }

    uint32_t nq = 0;
    (::vkGetPhysicalDeviceQueueFamilyProperties)(gpu, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qs(nq);
    (::vkGetPhysicalDeviceQueueFamilyProperties)(gpu, &nq, qs.data());
    uint32_t family = 0;
    for (uint32_t i = 0; i < nq; ++i)
        if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDevice(gpu, &dci, nullptr, &dev));

    // THE DISPATCH TABLE IS NOT RELOADED FOR THIS DEVICE. vkCmd* through the macros would go
    // to the ENGINE's device pointers, which is undefined behaviour on a second device -- so
    // the two calls in the loop below are written with (::vkName), the loader's own symbol.
    VkDescriptorSetLayoutBinding b = {};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo dsl = {};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &b;
    VkDescriptorSetLayout sl = VK_NULL_HANDLE;
    VK_CHECK((::vkCreateDescriptorSetLayout)(dev, &dsl, nullptr, &sl));

    VkDescriptorPoolSize size = {};
    size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    size.descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 2;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VK_CHECK((::vkCreateDescriptorPool)(dev, &dpi, nullptr, &dpool));

    const VkDescriptorSetLayout two[2] = { sl, sl };
    VkDescriptorSetAllocateInfo dai = {};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = dpool;
    dai.descriptorSetCount = 2;
    dai.pSetLayouts = two;
    VkDescriptorSet sets[2] = {};
    VK_CHECK((::vkAllocateDescriptorSets)(dev, &dai, sets));

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(Mat4) + 6 * sizeof(float) + 2 * sizeof(int32_t);
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 2;
    pli.pSetLayouts = two;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK((::vkCreatePipelineLayout)(dev, &pli, nullptr, &layout));

    VkCommandPoolCreateInfo cpi = {};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK((::vkCreateCommandPool)(dev, &cpi, nullptr, &pool));

    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK((::vkAllocateCommandBuffers)(dev, &cbi, &cmd));

    struct Push { Mat4 model; float tint[4]; int32_t material; } push = {};
    push.model = identity();

    auto run = [&](bool rebind) {
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK((::vkResetCommandBuffer)(cmd, 0));
        VK_CHECK((::vkBeginCommandBuffer)(cmd, &begin));
        const double t0 = now_us();
        for (int i = 0; i < DRAWS; ++i) {
            if (rebind)
                (::vkCmdBindDescriptorSets)(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                                            0, 2, sets, 0, nullptr);
            push.material = i % MATERIAL_COUNT;
            (::vkCmdPushConstants)(cmd, layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof push, &push);
        }
        const double us = now_us() - t0;
        VK_CHECK((::vkEndCommandBuffer)(cmd));
        return us;
    };

    // ---- AND TODAY'S TWO, ON THE SAME LAYERLESS DEVICE --------------------------------------
    // NOTHING IS SUBMITTED. This measures what it costs the CPU to RECORD the calls, which is
    // the whole of what instancing saves -- the card draws the same number of triangles either
    // way. A submitted version would be measuring the rasterizer instead, and answering a
    // different question.
    const int STARS = 10000;
    auto record_draws = [&](bool instanced) {
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK((::vkResetCommandBuffer)(cmd, 0));
        VK_CHECK((::vkBeginCommandBuffer)(cmd, &begin));
        const double t0 = now_us();
        if (instanced) {
            // TIMED OVER TEN THOUSAND CALLS AND DIVIDED, because ONE call is below the
            // resolution of the clock -- and a ratio whose denominator is a rounding error
            // moves by a factor of two between runs. Day 2 said measure the thing you think
            // you are measuring; this is the same mistake in a new place.
            for (int i = 0; i < STARS; ++i)
                (::vkCmdDrawIndexed)(cmd, 576, uint32_t(STARS), 0, 0, 0);
        } else {
            for (int i = 0; i < STARS; ++i) {
                // ONE PUSH AND ONE DRAW A COPY, which is what an engine without instancing
                // has to do: the per-object data has to reach the shader somehow.
                (::vkCmdPushConstants)(cmd, layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0, sizeof push, &push);
                (::vkCmdDrawIndexed)(cmd, 576, 1, 0, 0, 0);
            }
        }
        const double us = now_us() - t0;
        VK_CHECK((::vkEndCommandBuffer)(cmd));
        return us;
    };
    record_draws(false); record_draws(true);
    c.many_draws_us = record_draws(false);
    c.one_draw_us = record_draws(true) / double(STARS);

    // RECORDED TWICE AND THE SECOND ONE COUNTS: the first pass through a command pool pays for
    // allocation the second does not.
    run(false); run(true);
    c.bind_push_us = run(false);
    c.bind_set_us = run(true);

    std::printf("record   : %d draws, push only        %8.1f us  %6.1f ns each\n",
                DRAWS, c.bind_push_us, 1000.0 * c.bind_push_us / DRAWS);
    std::printf("record   : %d draws, rebinding sets   %8.1f us  %6.1f ns each\n",
                DRAWS, c.bind_set_us, 1000.0 * c.bind_set_us / DRAWS);
    std::printf("record   : the bind costs %.1f ns a draw, with NO layers on it\n",
                1000.0 * (c.bind_set_us - c.bind_push_us) / DRAWS);
    std::printf("instance : %d copies, one draw each %9.1f us  %5.1f ns a copy\n",
                STARS, c.many_draws_us, 1000.0 * c.many_draws_us / double(STARS));
    std::printf("instance : %d copies, ONE draw call %9.3f us  %5.1f ns the call\n",
                STARS, c.one_draw_us, 1000.0 * c.one_draw_us);
    std::printf("instance : the same triangles, recorded %.0f times more cheaply\n",
                c.many_draws_us / (c.one_draw_us > 0.0 ? c.one_draw_us : 1.0));

    (::vkFreeCommandBuffers)(dev, pool, 1, &cmd);
    (::vkDestroyCommandPool)(dev, pool, nullptr);
    (::vkDestroyPipelineLayout)(dev, layout, nullptr);
    (::vkDestroyDescriptorPool)(dev, dpool, nullptr);
    (::vkDestroyDescriptorSetLayout)(dev, sl, nullptr);
    (::vkDestroyDevice)(dev, nullptr);
    (::vkDestroyInstance)(inst, nullptr);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    const char* want = nullptr;
    bool provoke = false, window_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) want = argv[++i];
        else if (std::strcmp(argv[i], "--provoke") == 0) provoke = true;
        else if (std::strcmp(argv[i], "--window") == 0) window_mode = true;
    }
    // A full run always ends with the provocation, so one capture holds the whole day --
    // including the validation layer's account of a real mistake.
    if (!want) provoke = true;

    Context c;
    make_context(c, true);

    // ---- THE MEMORY TABLE THIS CARD ACTUALLY OFFERS ---------------------------------------
    // Not an UPLOAD heap and a DEFAULT heap. A list, with flags, and the choice is yours.
    VkPhysicalDeviceMemoryProperties mp = {};
    vkGetPhysicalDeviceMemoryProperties(c.gpu, &mp);
    std::printf("memory   : %u type(s) in %u heap(s)\n", mp.memoryTypeCount, mp.memoryHeapCount);
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        std::printf("  heap %u  %6llu MB  %s\n", i,
                    (unsigned long long)(mp.memoryHeaps[i].size / (1024 * 1024)),
                    (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                        ? "device local" : "host");
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        std::printf("  type %2u heap %u  %s%s%s%s\n", i, mp.memoryTypes[i].heapIndex,
                    (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                    (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED" : "");
    }

    // ---- THE BUFFERS, BUILT ONCE ----------------------------------------------------------
    make_geometry(c, 48, 64, false);
    for (int i = 0; i < MATERIAL_COUNT; ++i) make_texture(c, i);
    make_shared_sampler(c);
    make_descriptors(c);
    make_post(c);
    make_dust(c);
    c.dust_comp = make_dust_compute(c, DUST_LOCAL);
    c.dust_gfx = make_dust_pipeline(c, VK_FORMAT_R16G16B16A16_SFLOAT);
    make_stars(c);
    c.star_pipe = make_star_pipeline(c, VK_FORMAT_R16G16B16A16_SFLOAT);
    make_shadow_map(c, SCENES[0].map_size);
    write_shadow_descriptor(c);
    c.shadow_pipe = make_shadow_pipeline(c, false);
    c.shadow_pipe_front = make_shadow_pipeline(c, true);
    std::printf("surface  : 2 floats a part -- roughness and metalness, in the push block\n");
    for (const Part& pt : c.parts)
        std::printf("  %-9s roughness %.2f  metalness %.0f\n",
                    pt.name, pt.roughness, pt.metalness);
    std::printf("shadow   : %ux%u D32_SFLOAT, %llu bytes, comparison sampler\n",
                c.shadow_size, c.shadow_size, (unsigned long long)c.shadow_bytes);
    std::printf("shadow   : one vertex shader, no fragment shader, no colour attachment\n");
    std::printf("hdr      : R16G16B16A16_SFLOAT, %u bytes a pixel, %.1f MB at %ux%u\n",
                8u, 8.0 * RENDER_W * RENDER_H / 1048576.0, RENDER_W, RENDER_H);
    std::printf("bloom    : two %ux%u buffers, a quarter of the pixels each\n",
                RENDER_W / 2, RENDER_H / 2);
    std::printf("frame    : 5 passes -- dispatch, shadow, scene, bloom (x3), tone map\n");
    std::printf("star     : %u vertices, %u indices, %u triangles, one mesh\n",
                c.star_vertices, c.star_indices, c.star_indices / 3);
    std::printf("star     : %u instances of %u bytes -- %.2f MB, host visible\n",
                MAX_STARS, unsigned(sizeof(Star)),
                double(MAX_STARS) * sizeof(Star) / 1048576.0);
    std::printf("star     : binding 0 per vertex, binding 1 per INSTANCE\n");
    measure_dispatch(c);
    measure_pipelines(c);
    measure_binding(c);
    std::printf("memtype  : cube_vb in type %u, cube_ib in type %u\n",
                c.cube_vb.type_index, c.cube_ib.type_index);

    // ---- THE FLIP, AS A NUMBER RATHER THAN AN ASSERTION -----------------------------------
    // The chapter claims one element of the projection matrix changes sign. This prints it,
    // from the same function the renderer calls, so the claim cannot drift away from the code.
    const Mat4 up = view_proj_for(SCENES[0]);
    const Mat4 down = view_proj_for(SCENES[7]);
    (void)up; (void)down;

    // ---- THE CLAIM DAY 6 MADE, AS A NUMBER --------------------------------------------------
    // placement * view should be the identity, because the view matrix IS the placement
    // inverted. Anything but a rounding error here means the two were built from different
    // ideas of what the camera is.
    for (const Scene& sc : SCENES) {
        const Mat4 both = mul(camera_placement(sc.cam), camera_view(sc.cam));
        float fwd[3], right[3];
        camera_axes(sc.cam, fwd, right);
        std::printf("camera   : %-9s at %+5.1f %+4.1f %+5.1f  yaw %+5.2f pitch %+5.2f  "
                    "fwd %+4.2f %+4.2f %+4.2f\n",
                    sc.name, sc.cam.x, sc.cam.y, sc.cam.z, sc.cam.yaw, sc.cam.pitch,
                    fwd[0], fwd[1], fwd[2]);
        std::printf("inverse  : %-9s placement * view is %.2e from the identity\n",
                    sc.name, off_identity(both));
    }

    // ---- A WINDOW, IF ASKED FOR, AND THE WHOLE OF WHAT IT ADDS ---------------------------
    // It is created even for a capture, and NOT SHOWN. Creating a surface is what proves the
    // extensions work and what makes the capture report what this machine offers -- and a
    // window nobody sees cannot land in a screenshot, which is this book's rule.
    Window win;
    if (!make_window(c, win, 1280, 720, window_mode)) {
        std::printf("FAILED: no window or no surface\n");
        return 1;
    }
    if (!window_mode)
        std::printf("window   : created, never shown -- the capture opens no screen\n");

    if (window_mode) {
        // ONE FRAME, PRESENTED, AND THE FOUR STEPS EVERY VULKAN FRAME HAS.
        std::printf("window   : presenting until Esc\n");
        std::vector<uint8_t> px;
        bool sw = false;
        const Scene& s = SCENES[0];
        (void)render_scene(c, s, px, &sw);
        MSG msg = {};
        bool running = true;
        while (running) {
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) running = false;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!running) break;
            uint32_t index = 0;
            const VkResult acq = vkAcquireNextImageKHR(c.device, win.swapchain, UINT64_MAX,
                                                       win.acquired, VK_NULL_HANDLE, &index);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) break;
            VkPresentInfoKHR pi = {};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &win.acquired;
            pi.swapchainCount = 1;
            pi.pSwapchains = &win.swapchain;
            pi.pImageIndices = &index;
            if (vkQueuePresentKHR(c.queue, &pi) != VK_SUCCESS) break;
            vkQueueWaitIdle(c.queue);
        }
        vkDeviceWaitIdle(c.device);
        destroy_window(c, win);
        vkDestroyPipelineLayout(c.device, c.layout, nullptr);
        vkDestroyShaderModule(c.device, c.vs_mesh, nullptr);
        vkDestroyShaderModule(c.device, c.fs, nullptr);
        vkDestroyCommandPool(c.device, c.pool, nullptr);
        vkDestroyDevice(c.device, nullptr);
        vkDestroyInstance(c.instance, nullptr);
        CoUninitialize();
        return 0;
    }

    // THE TESSELLATION IS A PROPERTY OF THE SCENE, so a scene asking for a different one
    // gets a different mesh. This is the first time the engine destroys and rebuilds a
    // resource while running, which means the first vkDeviceWaitIdle outside teardown.
    bool swapped = false;
    int made = 0;
    std::vector<uint8_t> px;
    std::map<std::string, std::vector<uint8_t>> kept;
    for (const Scene& s : SCENES) {
        if (want && std::strcmp(want, s.name) != 0) continue;
        if (s.map_size != c.shadow_size) {
            vkDeviceWaitIdle(c.device);
            destroy_shadow_map(c);
            make_shadow_map(c, s.map_size);
            write_shadow_descriptor(c);
            std::printf("shadow   : rebuilt at %ux%u -- %llu bytes, 1 texel is %.4f units\n",
                        c.shadow_size, c.shadow_size,
                        (unsigned long long)c.shadow_bytes,
                        2.0f * s.sun_half / float(c.shadow_size));
        }
        if (!render_scene(c, s, px, &swapped)) {
            std::printf("FAILED: writing frame_%s.png\n", s.name);
            return 1;
        }
        // The frames are kept so the comparison below is made from the same bytes that
        // went into the PNG rather than from a second render.
        kept[s.name] = px;
        const Draw d = draw_for(c, s);
        (void)d;
        // THE LINE FITS EIGHTY-EIGHT COLUMNS, which is the width of a printed listing. A
        // capture line wider than the measure is a line this book cannot quote.
        std::printf("wrote  : %-7s %6d of %6d stars  %5.1f%% culled  %s\n",
                    s.name, c.drawn, s.stars,
                    s.stars ? 100.0 * double(c.culled) / double(s.stars) : 0.0, s.note);
        std::printf("cull   : %-7s %6d test(s) in %7.1f us -- %.1f ns each\n",
                    s.name, s.stars, c.cull_us,
                    s.stars ? 1000.0 * c.cull_us / double(s.stars) : 0.0);
        ++made;
    }

    // ---- WHAT READING THE SAME BYTES DIFFERENTLY COST -------------------------------------
    // Every scene is 1920x1080, so all of them compare pixel for pixel. The numbers the chapter
    // prints are these numbers, and if the engine changes they change.
    if (!want && !kept["stars"].empty()) {
        struct Pair { const char* a; const char* b; };
        static const Pair PAIRS[] = {
            { "stars",  "none"  },
            { "few",    "stars" },
            { "dense",  "stars" },
            { "big",    "stars" },
            { "glow",   "stars" },
            { "frozen", "stars" },
            { "wedge",  "wide"  },
            { "tilt",   "stars" },
        };
        for (const Pair& pr : PAIRS) {
            if (kept[pr.a].empty() || kept[pr.b].empty()) continue;
            std::printf("compare: %-9s vs %-9s %6.2f%% of pixels differ\n",
                        pr.a, pr.b, 100.0 * fraction_differing(kept[pr.a], kept[pr.b]));
        }
    }

    if (provoke) {
        // A DELIBERATE MISTAKE, so the layer can be seen doing its job. A command buffer that
        // is submitted twice without being reset breaks a rule the driver will not mention.
        std::printf("provoke  : submitting a command buffer that was never recorded\n");
        VkCommandBufferAllocateInfo cbi = {};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbi.commandPool = c.pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        VkCommandBuffer empty = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(c.device, &cbi, &empty);

        VkCommandBufferSubmitInfo csi = {};
        csi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        csi.commandBuffer = empty;
        VkSubmitInfo2 si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &csi;
        const VkResult r = vkQueueSubmit2(c.queue, 1, &si, VK_NULL_HANDLE);
        std::printf("provoke  : vkQueueSubmit2 returned %d, and the layer said the rest\n",
                    (int)r);
        vkQueueWaitIdle(c.queue);
        vkFreeCommandBuffers(c.device, c.pool, 1, &empty);
    }

    std::printf("wic      : asked for RGBA and was given %s\n", swapped ? "BGRA" : "RGBA");
    std::printf("debug    : %d message(s) from the validation layer this run\n", g_messages);
    std::printf("done     : %d picture(s), no window opened\n", made);

    destroy_geometry(c);
    for (int i = 0; i < MATERIAL_COUNT; ++i) {
        vkDestroyImageView(c.device, c.tex_view[i], nullptr);
        vkDestroyImage(c.device, c.tex[i], nullptr);
        vkFreeMemory(c.device, c.tex_mem[i], nullptr);
    }
    destroy_buffer(c, c.camera_ub);
    destroy_buffer(c, c.lights_sb);
    destroy_shadow_map(c);
    vkDestroyPipeline(c.device, c.star_pipe, nullptr);
    vkDestroyPipelineLayout(c.device, c.star_layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_star, nullptr);
    vkDestroyShaderModule(c.device, c.fs_star, nullptr);
    destroy_buffer(c, c.star_vb);
    destroy_buffer(c, c.star_ib);
    destroy_buffer(c, c.instances);
    vkDestroyPipeline(c.device, c.dust_comp, nullptr);
    vkDestroyPipeline(c.device, c.dust_gfx, nullptr);
    vkDestroyPipelineLayout(c.device, c.dust_comp_layout, nullptr);
    vkDestroyPipelineLayout(c.device, c.dust_gfx_layout, nullptr);
    vkDestroyDescriptorPool(c.device, c.dust_pool, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.dust_layout, nullptr);
    vkDestroyShaderModule(c.device, c.cs_dust, nullptr);
    vkDestroyShaderModule(c.device, c.vs_dust, nullptr);
    vkDestroyShaderModule(c.device, c.fs_dust, nullptr);
    destroy_buffer(c, c.motes);
    vkDestroySampler(c.device, c.post_sampler, nullptr);
    vkDestroyDescriptorPool(c.device, c.post_pool, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.post1_layout, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.post2_layout, nullptr);
    vkDestroyPipelineLayout(c.device, c.post1_pipe_layout, nullptr);
    vkDestroyPipelineLayout(c.device, c.post2_pipe_layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_post, nullptr);
    vkDestroyShaderModule(c.device, c.fs_bloom, nullptr);
    vkDestroyShaderModule(c.device, c.fs_tonemap, nullptr);
    vkDestroyPipeline(c.device, c.shadow_pipe, nullptr);
    vkDestroyPipeline(c.device, c.shadow_pipe_front, nullptr);
    vkDestroyPipelineLayout(c.device, c.shadow_layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_shadow, nullptr);
    vkDestroySampler(c.device, c.sampler, nullptr);
    vkDestroyDescriptorPool(c.device, c.pool_desc, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.camera_layout, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.material_layout, nullptr);
    if (c.cache) vkDestroyPipelineCache(c.device, c.cache, nullptr);
    destroy_window(c, win);
    vkDestroyPipelineLayout(c.device, c.layout, nullptr);
    vkDestroyShaderModule(c.device, c.vs_mesh, nullptr);
    vkDestroyShaderModule(c.device, c.fs, nullptr);
    vkDestroyCommandPool(c.device, c.pool, nullptr);
    vkDestroyDevice(c.device, nullptr);
    if (c.messenger) {
        auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(c.instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(c.instance, c.messenger, nullptr);
    }
    vkDestroyInstance(c.instance, nullptr);
    CoUninitialize();
    return 0;
}
