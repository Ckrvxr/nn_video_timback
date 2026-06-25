/**
 * c_rt_itctp_decoder — Decode MKV → ICtCp float32 via FFmpeg + AVX2 SIMD.
 *
 * Modes:
 *   --lr L.mkv --hr H.mkv --pipe
 *   --lr L.mkv --hr H.mkv --output f.npy
 *   --dataset <dir> --output <dir>
 *   --bench <file>
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <immintrin.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// ── BT.2100 PQ ───────────────────────────────────────────────────────
static constexpr float M1   = 2610.0f / 16384.0f;
static constexpr float M2   = 2523.0f / 32.0f;
static constexpr float C1   = 3424.0f / 4096.0f;
static constexpr float C2   = 2413.0f / 4096.0f * 32.0f;
static constexpr float C3   = 2392.0f / 4096.0f * 32.0f;
static constexpr float IM2  = 1.0f / M2;  // 32/2523
static constexpr float INVM1 = 1.0f / M1;

static const float M_YUV2RGB[9] = {1.0f, 0.0f, 1.4746f, 1.0f, -0.1646f, -0.5714f, 1.0f, 1.8814f, 0.0f};
static const float M_RGB2LMS[9] = {
    1688.0f/4096.0f, 2146.0f/4096.0f,  262.0f/4096.0f,
     683.0f/4096.0f, 2951.0f/4096.0f,  462.0f/4096.0f,
      99.0f/4096.0f,  309.0f/4096.0f, 3688.0f/4096.0f};
static const float M_LMS2ICTCP[9] = {
    2048.0f/4096.0f, 2048.0f/4096.0f,     0.0f,
    6610.0f/4096.0f,-13613.0f/4096.0f, 7003.0f/4096.0f,
   17933.0f/4096.0f,-17390.0f/4096.0f, -543.0f/4096.0f};

static constexpr int LUT_SZ = 1 << 16;
static float OETF_LUT[LUT_SZ + 1];

// 256³ 3D LUT: YUV→ICtCp (192 MB)
// idx = (v << 16 | u << 8 | y_index) * 3
static constexpr int LUT_Y = 256;
static std::vector<float> YUV_LUT8;

static void init_luts() {
    for (int i = 0; i < LUT_SZ; i++) {
        float v = float(i) / (LUT_SZ - 1);
        float x = std::pow(v, M1);
        OETF_LUT[i] = std::pow((C1 + C2*x) / (1.0f + C3*x), M2);
    }
    OETF_LUT[LUT_SZ] = OETF_LUT[LUT_SZ - 1];  // clamp
    // Precompute or load 256³ YUV→ICtCp
    const char *lut_path = "/tmp/ictcp_lut8.bin";
    FILE *f = fopen(lut_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz == 256L*256*256*3*4) {
            YUV_LUT8.resize(256*256*256*3);
            fread(YUV_LUT8.data(), 4, 256L*256*256*3, f);
            fclose(f); return;
        }
        fclose(f);
    }
    YUV_LUT8.resize(256 * 256 * 256 * 3);
    for (int v = 0; v < 256; v++) {
        for (int u = 0; u < 256; u++) {
            for (int ri = 0; ri < 256; ri++) {
                float yf = ri / 255.0f;
                float uf = (u - 128.0f) / 255.0f;
                float vf = (v - 128.0f) / 255.0f;
                float r = std::max(0.0f, std::min(1.0f, yf + vf*M_YUV2RGB[2]));
                float g = std::max(0.0f, std::min(1.0f, yf + uf*M_YUV2RGB[4] + vf*M_YUV2RGB[5]));
                float b = std::max(0.0f, std::min(1.0f, yf + uf*M_YUV2RGB[7]));
                r = std::pow(std::max(std::pow(r, IM2) - C1, 0.0f) / (C2 - C3*std::pow(r, IM2) + 1e-8f), INVM1);
                g = std::pow(std::max(std::pow(g, IM2) - C1, 0.0f) / (C2 - C3*std::pow(g, IM2) + 1e-8f), INVM1);
                b = std::pow(std::max(std::pow(b, IM2) - C1, 0.0f) / (C2 - C3*std::pow(b, IM2) + 1e-8f), INVM1);
                float l0 = r*M_RGB2LMS[0] + g*M_RGB2LMS[1] + b*M_RGB2LMS[2];
                float l1 = r*M_RGB2LMS[3] + g*M_RGB2LMS[4] + b*M_RGB2LMS[5];
                float l2 = r*M_RGB2LMS[6] + g*M_RGB2LMS[7] + b*M_RGB2LMS[8];
                l0 = std::min(std::max(l0, 0.0f), 1.0f); int i0 = (int)(l0 * (LUT_SZ-1)); l0 = OETF_LUT[i0] + (OETF_LUT[i0+1]-OETF_LUT[i0])*(l0*(LUT_SZ-1)-i0);
                l1 = std::min(std::max(l1, 0.0f), 1.0f); int i1 = (int)(l1 * (LUT_SZ-1)); l1 = OETF_LUT[i1] + (OETF_LUT[i1+1]-OETF_LUT[i1])*(l1*(LUT_SZ-1)-i1);
                l2 = std::min(std::max(l2, 0.0f), 1.0f); int i2 = (int)(l2 * (LUT_SZ-1)); l2 = OETF_LUT[i2] + (OETF_LUT[i2+1]-OETF_LUT[i2])*(l2*(LUT_SZ-1)-i2);
                int idx = ((v << 16) | (u << 8) | ri) * 3;
                YUV_LUT8[idx]   = l0*M_LMS2ICTCP[0] + l1*M_LMS2ICTCP[1] + l2*M_LMS2ICTCP[2];
                YUV_LUT8[idx+1] = l0*M_LMS2ICTCP[3] + l1*M_LMS2ICTCP[4] + l2*M_LMS2ICTCP[5];
                YUV_LUT8[idx+2] = l0*M_LMS2ICTCP[6] + l1*M_LMS2ICTCP[7] + l2*M_LMS2ICTCP[8];
            }
        }
    }
    // Save LUT to disk for future runs
    FILE *fw = fopen(lut_path, "wb");
    if (fw) { fwrite(YUV_LUT8.data(), 4, 256L*256*256*3, fw); fclose(fw); }
}

// ── Per-pixel EOTF (scalar, compiler auto-vectorizes pow) ────────────
static inline float eotf(float v) {
    v = std::min(std::max(v, 0.0f), 1.0f);
    float t = std::pow(v, IM2);
    return std::pow(std::max(t - C1, 0.0f) / (C2 - C3*t + 1e-8f), INVM1);
}

// ── 8-pixel AVX2 kernel: YUV444 → ICtCp ─────────────────────────────
// Reads 8× Y, U, V values (16-bit or 8-bit), writes 8× ICtCp interleaved
static void kernel_avx2(
    const uint16_t *py16, const uint16_t *pu16, const uint16_t *pv16,
    const uint8_t  *py8,  const uint8_t  *pu8,  const uint8_t  *pv8,
    float *out, bool is_16bit,
    float peak, float half)
{
    __m256 yf, uf, vf;
    if (is_16bit) {
        __m256i yv = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)py16));
        __m256i uv = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)pu16));
        __m256i vv = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)pv16));
        yf = _mm256_cvtepi32_ps(yv); uf = _mm256_cvtepi32_ps(uv); vf = _mm256_cvtepi32_ps(vv);
    } else {
        __m128i y8 = _mm_loadl_epi64((__m128i*)py8);
        __m128i u8 = _mm_loadl_epi64((__m128i*)pu8);
        __m128i v8 = _mm_loadl_epi64((__m128i*)pv8);
        yf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(y8));
        uf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8));
        vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v8));
    }
    // Normalize: y/=peak, (u-half)/peak, (v-half)/peak
    __m256 pinv = _mm256_set1_ps(1.0f/peak);
    yf = _mm256_mul_ps(yf, pinv);
    uf = _mm256_fmadd_ps(uf, pinv, _mm256_set1_ps(-half/peak));
    vf = _mm256_fmadd_ps(vf, pinv, _mm256_set1_ps(-half/peak));

    // YUV → RGB  (BT.2020)
    __m256 r = _mm256_add_ps(yf, _mm256_mul_ps(vf, _mm256_set1_ps( M_YUV2RGB[2])));
    __m256 g = _mm256_add_ps(
        _mm256_add_ps(yf, _mm256_mul_ps(uf, _mm256_set1_ps(M_YUV2RGB[4]))),
        _mm256_mul_ps(vf, _mm256_set1_ps(M_YUV2RGB[5])));
    __m256 b = _mm256_add_ps(yf, _mm256_mul_ps(uf, _mm256_set1_ps(M_YUV2RGB[7])));

    // Store for scalar EOTF + LMS + OETF
    float R[8], G[8], B[8];
    _mm256_storeu_ps(R, r); _mm256_storeu_ps(G, g); _mm256_storeu_ps(B, b);

        for (int i = 0; i < 8; i++) {
        float rl = eotf(R[i]), gl = eotf(G[i]), bl = eotf(B[i]);
        float l0 = rl*M_RGB2LMS[0] + gl*M_RGB2LMS[1] + bl*M_RGB2LMS[2];
        float l1 = rl*M_RGB2LMS[3] + gl*M_RGB2LMS[4] + bl*M_RGB2LMS[5];
        float l2 = rl*M_RGB2LMS[6] + gl*M_RGB2LMS[7] + bl*M_RGB2LMS[8];
        l0 = std::min(std::max(l0, 0.0f), 1.0f); int i0 = int(l0 * LUT_SZ); float f0 = l0*LUT_SZ - i0;
        l1 = std::min(std::max(l1, 0.0f), 1.0f); int i1 = int(l1 * LUT_SZ); float f1 = l1*LUT_SZ - i1;
        l2 = std::min(std::max(l2, 0.0f), 1.0f); int i2 = int(l2 * LUT_SZ); float f2 = l2*LUT_SZ - i2;
        l0 = OETF_LUT[i0] + (OETF_LUT[i0+1]-OETF_LUT[i0])*f0; l1 = OETF_LUT[i1] + (OETF_LUT[i1+1]-OETF_LUT[i1])*f1; l2 = OETF_LUT[i2] + (OETF_LUT[i2+1]-OETF_LUT[i2])*f2;
        out[i*3]   = l0*M_LMS2ICTCP[0] + l1*M_LMS2ICTCP[1] + l2*M_LMS2ICTCP[2];
        out[i*3+1] = l0*M_LMS2ICTCP[3] + l1*M_LMS2ICTCP[4] + l2*M_LMS2ICTCP[5];
        out[i*3+2] = l0*M_LMS2ICTCP[6] + l1*M_LMS2ICTCP[7] + l2*M_LMS2ICTCP[8];
    }
}

static void scalar_pixel(float y, float u, float v, float *out, float peak, float half) {
    y /= peak; u = u/peak - half/peak; v = v/peak - half/peak;
    float r = y + v*M_YUV2RGB[2];
    float g = y + u*M_YUV2RGB[4] + v*M_YUV2RGB[5];
    float b = y + u*M_YUV2RGB[7];
    float rl = eotf(r), gl = eotf(g), bl = eotf(b);
    float l0 = rl*M_RGB2LMS[0] + gl*M_RGB2LMS[1] + bl*M_RGB2LMS[2];
    float l1 = rl*M_RGB2LMS[3] + gl*M_RGB2LMS[4] + bl*M_RGB2LMS[5];
    float l2 = rl*M_RGB2LMS[6] + gl*M_RGB2LMS[7] + bl*M_RGB2LMS[8];
    l0 = std::min(std::max(l0,0.0f),1.0f); l1 = std::min(std::max(l1,0.0f),1.0f); l2 = std::min(std::max(l2,0.0f),1.0f);
    int i0=int(l0*LUT_SZ),i1=int(l1*LUT_SZ),i2=int(l2*LUT_SZ);
    float f0=l0*LUT_SZ-i0,f1=l1*LUT_SZ-i1,f2=l2*LUT_SZ-i2;
    l0=OETF_LUT[i0]+(OETF_LUT[i0+1]-OETF_LUT[i0])*f0; l1=OETF_LUT[i1]+(OETF_LUT[i1+1]-OETF_LUT[i1])*f1; l2=OETF_LUT[i2]+(OETF_LUT[i2+1]-OETF_LUT[i2])*f2;
    out[0]=l0*M_LMS2ICTCP[0]+l1*M_LMS2ICTCP[1]+l2*M_LMS2ICTCP[2];
    out[1]=l0*M_LMS2ICTCP[3]+l1*M_LMS2ICTCP[4]+l2*M_LMS2ICTCP[5];
    out[2]=l0*M_LMS2ICTCP[6]+l1*M_LMS2ICTCP[7]+l2*M_LMS2ICTCP[8];
}

// ── FFmpeg decoder → ICtCp ───────────────────────────────────────────

struct Frames { int H=0, W=0, N=0; std::vector<float> data; };

static int get_bits(AVCodecContext *ctx) {
    auto *d = av_pix_fmt_desc_get(ctx->pix_fmt);
    if (!d) return 8; int b = 0;
    for (int i = 0; i < d->nb_components; i++) b = std::max(b, d->comp[i].depth);
    return b ? b : 8;
}

Frames decode(const std::string &path) {
    Frames res;
    AVFormatContext *fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) return res;
    if (avformat_find_stream_info(fmt, nullptr) < 0) { avformat_close_input(&fmt); return res; }
    int si = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++)
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { si = i; break; }
    if (si < 0) { avformat_close_input(&fmt); return res; }
    auto *par = fmt->streams[si]->codecpar;
    auto *dec = avcodec_find_decoder(par->codec_id);
    if (!dec) { avformat_close_input(&fmt); return res; }
    AVCodecContext *ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(ctx, par);
    if (avcodec_open2(ctx, dec, nullptr) < 0) { avcodec_free_context(&ctx); avformat_close_input(&fmt); return res; }

    int W = ctx->width, H = ctx->height, bits = get_bits(ctx);
    float peak = (float)((1 << bits) - 1);
    float half = (float)(1 << (bits - 1));
    bool is16 = bits > 8;
    AVPixelFormat out_fmt = is16 ? AV_PIX_FMT_YUV444P12LE : AV_PIX_FMT_YUV444P;

    SwsContext *sws = sws_getContext(W, H, ctx->pix_fmt, W, H, out_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
    AVFrame *yuv444 = av_frame_alloc();
    yuv444->format = out_fmt; yuv444->width = W; yuv444->height = H;
    av_frame_get_buffer(yuv444, 32);

    std::vector<float> buf(H * W * 3);
    AVPacket *pkt = av_packet_alloc();
    AVFrame *fr = av_frame_alloc();

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != si) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(ctx, pkt) < 0) { av_packet_unref(pkt); break; }
        av_packet_unref(pkt);
        while (avcodec_receive_frame(ctx, fr) >= 0) {
            sws_scale(sws, fr->data, fr->linesize, 0, H, yuv444->data, yuv444->linesize);

            int shift = is16 ? bits - 8 : 0;
            for (int y = 0; y < H; y++) {
                float *pout = buf.data() + y * W * 3;
                int x = 0;
                if (is16) {
                    auto *py = (const uint16_t*)(yuv444->data[0] + y * yuv444->linesize[0]);
                    auto *pu = (const uint16_t*)(yuv444->data[1] + y * yuv444->linesize[1]);
                    auto *pv = (const uint16_t*)(yuv444->data[2] + y * yuv444->linesize[2]);
                    const float scale = 1.0f / (float)(1 << shift);
                    for (; x + 2 <= W; x += 2) {
                        for (int lane = 0; lane < 2; lane++) {
                            int yi = py[x+lane] >> shift, ui = pu[x+lane] >> shift, vi = pv[x+lane] >> shift;
                            float fy = (py[x+lane] - (yi << shift)) * scale;
                            float fu = (pu[x+lane] - (ui << shift)) * scale;
                            float fv = (pv[x+lane] - (vi << shift)) * scale;
                            int y1 = std::min(yi+1, 255), u1 = std::min(ui+1, 255), v1 = std::min(vi+1, 255);
                            auto l = [&](int yy, int uu, int vv) { return &YUV_LUT8[((vv<<16)|(uu<<8)|yy)*3]; };
                            float *c000=l(yi,ui,vi),*c001=l(yi,ui,v1),*c010=l(yi,u1,vi),*c011=l(yi,u1,v1);
                            float *c100=l(y1,ui,vi),*c101=l(y1,ui,v1),*c110=l(y1,u1,vi),*c111=l(y1,u1,v1);
                            for (int c = 0; c < 3; c++) {
                                float v00=c000[c]*(1-fv)+c001[c]*fv, v01=c010[c]*(1-fv)+c011[c]*fv;
                                float v10=c100[c]*(1-fv)+c101[c]*fv, v11=c110[c]*(1-fv)+c111[c]*fv;
                                float v0=v00*(1-fu)+v01*fu, v1=v10*(1-fu)+v11*fu;
                                pout[(x+lane)*3+c] = v0*(1-fy)+v1*fy;
                            }
                        }
                    }
                    for (; x < W; x++) {
                        int yi = py[x] >> shift, ui = pu[x] >> shift, vi = pv[x] >> shift;
                        float fy = (py[x] - (yi << shift)) * scale;
                        float fu = (pu[x] - (ui << shift)) * scale;
                        float fv = (pv[x] - (vi << shift)) * scale;
                        int y1=std::min(yi+1,255), u1=std::min(ui+1,255), v1=std::min(vi+1,255);
                        auto l=[&](int yy,int uu,int vv){return &YUV_LUT8[((vv<<16)|(uu<<8)|yy)*3];};
                        float *c000=l(yi,ui,vi),*c001=l(yi,ui,v1),*c010=l(yi,u1,vi),*c011=l(yi,u1,v1);
                        float *c100=l(y1,ui,vi),*c101=l(y1,ui,v1),*c110=l(y1,u1,vi),*c111=l(y1,u1,v1);
                        for (int c = 0; c < 3; c++) {
                            float v00=c000[c]*(1-fv)+c001[c]*fv, v01=c010[c]*(1-fv)+c011[c]*fv;
                            float v10=c100[c]*(1-fv)+c101[c]*fv, v11=c110[c]*(1-fv)+c111[c]*fv;
                            float v0=v00*(1-fu)+v01*fu, v1=v10*(1-fu)+v11*fu;
                            pout[x*3+c] = v0*(1-fy)+v1*fy;
                        }
                    }
                } else {
                    auto *py = yuv444->data[0] + y * yuv444->linesize[0];
                    auto *pu = yuv444->data[1] + y * yuv444->linesize[1];
                    auto *pv = yuv444->data[2] + y * yuv444->linesize[2];
                    for (; x + 4 <= W; x += 4) {
                        int idx0 = ((pv[x] << 16) | (pu[x] << 8) | py[x]) * 3;
                        int idx1 = ((pv[x+1] << 16) | (pu[x+1] << 8) | py[x+1]) * 3;
                        int idx2 = ((pv[x+2] << 16) | (pu[x+2] << 8) | py[x+2]) * 3;
                        int idx3 = ((pv[x+3] << 16) | (pu[x+3] << 8) | py[x+3]) * 3;
                        pout[x*3]=YUV_LUT8[idx0]; pout[x*3+1]=YUV_LUT8[idx0+1]; pout[x*3+2]=YUV_LUT8[idx0+2];
                        pout[(x+1)*3]=YUV_LUT8[idx1]; pout[(x+1)*3+1]=YUV_LUT8[idx1+1]; pout[(x+1)*3+2]=YUV_LUT8[idx1+2];
                        pout[(x+2)*3]=YUV_LUT8[idx2]; pout[(x+2)*3+1]=YUV_LUT8[idx2+1]; pout[(x+2)*3+2]=YUV_LUT8[idx2+2];
                        pout[(x+3)*3]=YUV_LUT8[idx3]; pout[(x+3)*3+1]=YUV_LUT8[idx3+1]; pout[(x+3)*3+2]=YUV_LUT8[idx3+2];
                    }
                    for (; x < W; x++) {
                        int idx = ((pv[x] << 16) | (pu[x] << 8) | py[x]) * 3;
                        pout[x*3]=YUV_LUT8[idx]; pout[x*3+1]=YUV_LUT8[idx+1]; pout[x*3+2]=YUV_LUT8[idx+2];
                    }
                }
            }
            res.data.insert(res.data.end(), buf.begin(), buf.end());
            res.N++;
        }
    }

    av_frame_free(&fr); av_frame_free(&yuv444); sws_freeContext(sws);
    av_packet_free(&pkt); avcodec_free_context(&ctx); avformat_close_input(&fmt);
    res.H = H; res.W = W;
    return res;
}

// ── Interleave ───────────────────────────────────────────────────────

static std::vector<float> interleave(const Frames &a, const Frames &b) {
    int N = std::min(a.N, b.N), s = a.H * a.W;
    std::vector<float> out(N * s * 6);
    for (int i = 0; i < N; i++)
        for (int p = 0; p < s; p++)
            for (int c = 0; c < 3; c++) {
                out[i*s*6 + p*6 + c]     = a.data[i*s*3 + p*3 + c];
                out[i*s*6 + p*6 + 3 + c] = b.data[i*s*3 + p*3 + c];
            }
    return out;
}

// ── I/O ──────────────────────────────────────────────────────────────

static void write_npy(const std::string &path, const float *d, int N, int H, int W, int C) {
    std::ofstream f(path, std::ios::binary);
    char hdr[256]; int len = std::snprintf(hdr,256,"{'descr':'<f4','fortran_order':False,'shape':(%d,%d,%d,%d),}",N,H,W,C);
    int pad = (16-(10+len)%16)%16; std::memset(hdr+len,' ',pad); len+=pad; hdr[len++]='\n';
    uint8_t mag[]={0x93,'N','U','M','P','Y',1,0}; uint16_t hl=len;
    f.write((char*)mag,8); f.write((char*)&hl,2); f.write(hdr,len);
    int64_t nf=(int64_t)N*H*W*C; f.write((char*)d,nf*4);
}

static void write_pipe(const float *d, int N, int H, int W) {
    int32_t h[3]={N,H,W}; fwrite(h,12,1,stdout);
    fwrite(d,4,(int64_t)N*H*W*6,stdout); fflush(stdout);
}

// ── Clip discovery ───────────────────────────────────────────────────

struct Clip { std::string lr, hr, name; };
static std::vector<Clip> discover(const std::string &dir) {
    std::vector<Clip> clips; DIR *d=opendir(dir.c_str());
    if(!d)return clips; struct dirent *e;
    while((e=readdir(d))){if(e->d_type!=DT_DIR)continue;
        std::string n=e->d_name; if(n=="."||n=="..")continue;
        std::string b=dir+"/"+n; struct stat st;
        if(stat((b+"/LR.mkv").c_str(),&st)==0&&S_ISREG(st.st_mode)&&
           stat((b+"/HR.mkv").c_str(),&st)==0&&S_ISREG(st.st_mode))
            clips.push_back({b+"/LR.mkv",b+"/HR.mkv",n});}
    closedir(d); return clips;
}

// ── Main ─────────────────────────────────────────────────────────────

// ── C-compatible exports for Python ctypes ──────────────────────────
extern "C" {

static thread_local std::vector<float> _buf;

void c_rt_itctp_init(void) { init_luts(); }

float* c_rt_itctp_decode(const char *lr_path, const char *hr_path,
                       int *out_N, int *out_H, int *out_W) {
    auto lf = decode(lr_path);
    auto hf = decode(hr_path);
    if (lf.N == 0 || hf.N == 0) return nullptr;
    int N = std::min(lf.N, hf.N), H = lf.H, W = lf.W;
    // Interleave in-place into _buf
    int stride = H * W;
    _buf.resize(N * stride * 6);
    for (int i = 0; i < N; i++)
        for (int p = 0; p < stride; p++)
            for (int c = 0; c < 3; c++) {
                _buf[i*stride*6 + p*6 + c]     = lf.data[i*stride*3 + p*3 + c];
                _buf[i*stride*6 + p*6 + 3 + c] = hf.data[i*stride*3 + p*3 + c];
            }
    *out_N = N; *out_H = H; *out_W = W;
    return _buf.data();
}

void c_rt_itctp_free(float *ptr) { (void)ptr; }

}  // extern "C"

// ── CLI ──────────────────────────────────────────────────────────────
#ifndef AS_LIB
int main(int argc, char **argv) {
    init_luts();
    std::string lr, hr, out, dataset; bool pipe=false, bench=false, daemon=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--lr")&&i+1<argc)lr=argv[++i];
        else if(!strcmp(argv[i],"--hr")&&i+1<argc)hr=argv[++i];
        else if(!strcmp(argv[i],"--output")&&i+1<argc)out=argv[++i];
        else if(!strcmp(argv[i],"--dataset")&&i+1<argc)dataset=argv[++i];
        else if(!strcmp(argv[i],"--pipe"))pipe=true;
        else if(!strcmp(argv[i],"--daemon"))daemon=true;
        else if(!strcmp(argv[i],"--bench")&&i+1<argc){bench=true;lr=argv[++i];}
        else {fprintf(stderr,"Unknown: %s\n",argv[i]);return 1;}
    }
    // Daemon mode: read LR/HR paths from stdin, write pipe data to stdout
    if(daemon){
        std::string line;
        while(std::getline(std::cin, lr)){
            if(!std::getline(std::cin, hr)) break;
            auto lf=decode(lr),hf=decode(hr);
            if(lf.N==0||hf.N==0) continue;
            auto d=interleave(lf,hf);
            write_pipe(d.data(),std::min(lf.N,hf.N),lf.H,lf.W);
        }
        return 0;
    }
    if(bench){
        auto t0=clock();auto f=decode(lr);double t=double(clock()-t0)/CLOCKS_PER_SEC;
        fprintf(stderr,"%s: %d frames (%dx%d) in %.3fs = %.0f fps\n",lr.c_str(),f.N,f.W,f.H,t,f.N/t);
        return 0;
    }
    if(!lr.empty()&&!hr.empty()){
        auto lf=decode(lr),hf=decode(hr);
        if(lf.N==0||hf.N==0)return 1;
        auto d=interleave(lf,hf);
        if(pipe)write_pipe(d.data(),std::min(lf.N,hf.N),lf.H,lf.W);
        else if(!out.empty())write_npy(out,d.data(),std::min(lf.N,hf.N),lf.H,lf.W,6);
        return 0;
    }
    if(!dataset.empty()&&!out.empty()){
        mkdir(out.c_str(),0755);
        for(auto&c:discover(dataset)){
            std::string np=out+"/"+c.name+".npy"; struct stat st;
            if(stat(np.c_str(),&st)==0)continue;
            fprintf(stderr,"  %s\n",c.name.c_str());
            auto lf=decode(c.lr),hf=decode(c.hr);
            write_npy(np,interleave(lf,hf).data(),std::min(lf.N,hf.N),lf.H,lf.W,6);
        }
        return 0;
    }
    fprintf(stderr,"Usage:\n"
                    "  c_rt_itctp_decoder --lr L.mkv --hr H.mkv --pipe\n"
                    "  c_rt_itctp_decoder --lr L.mkv --hr H.mkv --output f.npy\n"
                    "  c_rt_itctp_decoder --dataset <dir> --output <dir>\n"
                    "  c_rt_itctp_decoder --bench <file>\n"
                    "  c_rt_itctp_decoder --daemon\n");
    return 1;
}
#endif  // AS_LIB
