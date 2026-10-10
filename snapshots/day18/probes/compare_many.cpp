// compare_many.cpp -- what per cent of one engine frame differs from another
#include "png_probe.h"

// Every dust scene in this chapter, against the one it is meant to be judged by.
static const char* PAIRS[][2] = {
    { "many", "dust" },
    { "few", "dust" },
    { "big", "dust" },
    { "swirl", "dust" },
    { "fall", "dust" },
    { "late", "swirl" },
};

int main() {
    PngProbe p = png_open();
    for (const auto& pair : PAIRS) {
        UINT aw = 0, ah = 0, bw = 0, bh = 0;
        std::vector<uint8_t> a = png_load(p, pair[0], aw, ah);
        std::vector<uint8_t> b = png_load(p, pair[1], bw, bh);
        std::printf("compare: %-9s vs %-9s %6.2f%% of pixels differ\n",
                    pair[0], pair[1], png_differ(a, b));
    }
    png_close(p);
    return 0;
}
