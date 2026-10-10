// png_probe.h -- read the engine's own frame_*.png files back off disk and compare them.
//
// This header is the plumbing every picture-comparison probe in this book needs: WIC start-up,
// a PNG decode into 32-bit BGRA, and the pixel count. It is deliberately not printed in the
// chapter, so a probe that uses it stays inside this book's 25-line listing measure. It uses
// no graphics card at all -- the frames were written by the engine, and this only reads them.
#pragma once
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

struct PngProbe { ComPtr<IWICImagingFactory> wic; };

static void png_check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::printf("FAILED: %s  (hr=0x%08lX)\n", what, (unsigned long)hr);
        std::exit(1);
    }
}

static PngProbe png_open() {
    PngProbe p;
    png_check(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "CoInitializeEx");
    png_check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                               IID_PPV_ARGS(&p.wic)), "CoCreateInstance(WIC)");
    return p;
}

// A scene name, not a path: the engine writes frame_<scene>.png beside its own executable,
// which is the directory above this one.
static std::vector<uint8_t> png_load(PngProbe& p, const char* scene, UINT& w, UINT& h) {
    std::string path = std::string("../frame_") + scene + ".png";
    std::wstring wide(path.begin(), path.end());
    ComPtr<IWICBitmapDecoder> dec;
    png_check(p.wic->CreateDecoderFromFilename(wide.c_str(), nullptr, GENERIC_READ,
                                               WICDecodeMetadataCacheOnLoad, &dec),
              "CreateDecoderFromFilename");
    ComPtr<IWICBitmapFrameDecode> frame;
    png_check(dec->GetFrame(0, &frame), "GetFrame");
    ComPtr<IWICFormatConverter> conv;
    png_check(p.wic->CreateFormatConverter(&conv), "CreateFormatConverter");
    png_check(conv->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
                               WICBitmapDitherTypeNone, nullptr, 0.0,
                               WICBitmapPaletteTypeCustom), "Initialize");
    png_check(conv->GetSize(&w, &h), "GetSize");
    std::vector<uint8_t> px(size_t(w) * h * 4);
    png_check(conv->CopyPixels(nullptr, w * 4, (UINT)px.size(), px.data()), "CopyPixels");
    return px;
}

// Alpha is ignored on purpose: the picture is what you can see.
static double png_differ(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return 100.0;
    size_t n = a.size() / 4, differ = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = &a[i * 4];
        const uint8_t* q = &b[i * 4];
        if (p[0] != q[0] || p[1] != q[1] || p[2] != q[2]) ++differ;
    }
    return 100.0 * double(differ) / double(n);
}

// The factory has to be released BEFORE COM is shut down. Letting the ComPtr run its
// destructor after CoUninitialize releases an interface into a dead apartment, which
// crashes rather than failing, and does it after the output has already been printed.
static void png_close(PngProbe& p) {
    p.wic.Reset();
    CoUninitialize();
}
