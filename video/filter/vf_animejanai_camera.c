/*
 * Camera-mode transition metric for vf_animejanai's re-timing (see
 * vf_animejanai_camera.h). Every float operation below mirrors what
 * numpy 2.3.4 does for anime_retime.py (commit d28fca0) on x86-64 with
 * AVX2+FMA3 or AVX-512, operation for operation; the comments name the
 * numpy step. Measured, not assumed (camera_metric parity harness):
 *  - rfft2 of a float32 plane runs pocketfft in double (np.fft passes the
 *    Python int 1 as the norm factor, which selects the float64 loop) and
 *    stores complex64 after each axis; irfft2 runs in float32 (its factor is
 *    an np.float32 reciprocal);
 *  - complex64 multiply and absolute use fused multiply-adds (numpy's
 *    AVX2/AVX-512 loops): re = fma(ar, br, -(ai*bi)), im = fma(ar, bi,
 *    ai*br), |z| = max * sqrt(fma(r, r, 1)) with r = min/max; complex64 /
 *    float32 multiplies by the reciprocal. The reference is the worker image
 *    run under vspipe: a Python build where numpy's temporary elision turns
 *    fa * conj(fb) into an in-place multiply swaps the operands of the
 *    imaginary part's fma (see norm_cross);
 *  - float32 means use numpy's pairwise summation over the whole
 *    contiguous array; 5x5 block means add each block row left to right,
 *    then the rows top to bottom.
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

// Contraction would fuse operations numpy keeps separate (GCC's default in
// ISO C mode, which mpv uses, is already off).
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vf_animejanai_camera.h"
#include "vf_animejanai_camera_fft.h"

#define RC_N (RC_W * RC_H)
#define RC_FW (RC_W / 2 + 1)    // rfft2 columns
#define RC_SN (RC_H * RC_FW)    // spectrum values

struct rc_ctx {
    float *win;                 // hann_window(), RC_N
    struct rcf_rplan_d rfd;     // forward: double, like numpy
    struct rcf_cplan_d cfd;
    struct rcf_cplan_f cff;     // inverse: float32
    struct rcf_rplan_f rff;
};

struct rc_frame {
    float *plane;               // RC_N
    float *spec;                // RC_SN complex (re, im)
    bool has_plane, has_spec;
};

struct rc_work {
    rcf_vd *rd, *rd_s;          // RC_W
    rcf_cvd *cd, *cd_s;         // RC_H
    rcf_vf *rf, *rf_s;          // RC_W
    rcf_cvf *cf, *cf_s;         // RC_H
    float *q;                   // RC_SN complex: inverse column pass
    float *blk;                 // residual block differences
    uint64_t *acc;              // RC_N: non-multiple plane sums
    uint64_t *row;              // RC_W
};

// ---------------------------------------------------------------------------
// numpy float32 primitives
// ---------------------------------------------------------------------------

// numpy's pairwise summation (FLOAT_pairwise_sum, PW_BLOCKSIZE 128) of a
// contiguous float32 array.
static float pairwise_sum(const float *a, size_t n)
{
    if (n < 8) {
        float res = -0.0f;
        for (size_t i = 0; i < n; i++)
            res += a[i];
        return res;
    } else if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; j++)
            r[j] = a[j];
        size_t i;
        for (i = 8; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; j++)
                r[j] += a[i + j];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) +
                    ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++)
            res += a[i];
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

// x.mean() of a contiguous float32 array: add.reduce (identity, then the
// pairwise sum), then a division numpy does in double (np.intp count),
// which equals the float32 division for float32 operands.
static float np_mean(const float *a, size_t n)
{
    float s = 0.0f + pairwise_sum(a, n);
    return (float)((double)s / (double)n);
}

// Correctly rounded fused multiply-add, the result of numpy's FMA3 loops.
// Without hardware FMA: a*b is exact in double; round a*b + c to odd in
// double (TwoSum gives the exact remainder), then to nearest float, which
// is the correctly rounded result since 53 >= 24 + 2 (Boldo-Melquiond).
static inline float rc_fmaf(float a, float b, float c)
{
#if defined(__FMA__) || defined(__FMA4__)
    return __builtin_fmaf(a, b, c);
#else
    double p = (double)a * (double)b;
    double s = p + (double)c;
    double bp = s - p;
    double e = (p - (s - bp)) + ((double)c - bp);
    uint64_t u;
    memcpy(&u, &s, sizeof(u));
    if (e != 0 && !(u & 1)) {
        if ((e > 0) == (s > 0))
            u++;
        else
            u--;
        memcpy(&s, &u, sizeof(u));
    }
    return (float)s;
#endif
}

// r = fa * conj(fb); r / (|r| + 1e-6) for one bin (shift_from_spectra).
static inline void norm_cross(float ar, float ai, float br, float bi,
                              float *nr, float *ni)
{
    bi = -bi;  // np.conj
    // complex64 multiply, numpy's contiguous SIMD loop (simd_cmul_f32:
    // muladdsub(a_re, b, a_im * b_rev)), which is what the worker runs
    // (measured in vspipe, in ModifyFrame callbacks). The operand order of
    // the imaginary part matters: where numpy's temporary elision rewrites
    // `fa * np.conj(fb)` as an in-place multiply into the conj temporary
    // (it does under some Python builds, e.g. a shared-libpython venv), the
    // operands swap and im becomes fma(ai, br, ar*bi) instead.
    float re = rc_fmaf(ar, br, -(ai * bi));
    float im = rc_fmaf(ar, bi, ai * br);
    // complex64 absolute (simd_cabsolute_f32); inputs are finite here
    float x = fabsf(re), y = fabsf(im);
    float larger = x > y ? x : y, smaller = x > y ? y : x;
    float ratio = larger == 0.0f || smaller == INFINITY ? 0.0f
                                                         : smaller / larger;
    float hyp = sqrtf(rc_fmaf(ratio, ratio, 1.0f)) * larger;
    // + 1e-6 (a weak Python float: float32(1e-6))
    float den = hyp + (float)1e-6;
    // complex64 / float32 promotes to complex division (Smith's rule with a
    // zero imaginary part): rat = 0, scl = 1 / den
    float rat = 0.0f / den, scl = 1.0f / den;
    *nr = (re + im * rat) * scl;
    *ni = (im - re * rat) * scl;
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

// np.hanning(m): 0.5 + 0.5 * cos(pi * n / (m - 1)), n = 1-m, 3-m, ..., m-1
static void hanning(int m, double *out)
{
    for (int k = 0; k < m; k++) {
        volatile double a = 3.141592653589793 * (double)(1 - m + 2 * k);
        volatile double b = a / (double)(m - 1);
        volatile double c = 0.5 * cos(b);
        out[k] = 0.5 + c;
    }
}

struct rc_ctx *rc_ctx_create(void)
{
    struct rc_ctx *ctx = calloc(1, sizeof(*ctx));
    double *hy = malloc(RC_H * sizeof(double));
    double *hx = malloc(RC_W * sizeof(double));
    if (!ctx || !hy || !hx)
        goto fail;
    ctx->win = rcf_alloc(RC_N * sizeof(float));
    if (!ctx->win ||
        !rcf_rplan_init_d(&ctx->rfd, RC_W) ||
        !rcf_cplan_init_d(&ctx->cfd, RC_H) ||
        !rcf_cplan_init_f(&ctx->cff, RC_H) ||
        !rcf_rplan_init_f(&ctx->rff, RC_W))
        goto fail;
    hanning(RC_H, hy);
    hanning(RC_W, hx);
    for (int y = 0; y < RC_H; y++) {
        for (int x = 0; x < RC_W; x++) {
            volatile double p = hy[y] * hx[x];  // np.outer, then float32
            ctx->win[y * RC_W + x] = (float)p;
        }
    }
    free(hy);
    free(hx);
    return ctx;
fail:
    free(hy);
    free(hx);
    rc_ctx_destroy(ctx);
    return NULL;
}

void rc_ctx_destroy(struct rc_ctx *ctx)
{
    if (!ctx)
        return;
    rcf_free(ctx->win);
    rcf_rplan_uninit_d(&ctx->rfd);
    rcf_cplan_uninit_d(&ctx->cfd);
    rcf_cplan_uninit_f(&ctx->cff);
    rcf_rplan_uninit_f(&ctx->rff);
    free(ctx);
}

struct rc_work *rc_work_create(void)
{
    struct rc_work *wk = calloc(1, sizeof(*wk));
    if (!wk)
        return NULL;
    wk->rd = rcf_alloc(RC_W * sizeof(rcf_vd));
    wk->rd_s = rcf_alloc(RC_W * sizeof(rcf_vd));
    wk->cd = rcf_alloc(RC_H * sizeof(rcf_cvd));
    wk->cd_s = rcf_alloc(RC_H * sizeof(rcf_cvd));
    wk->rf = rcf_alloc(RC_W * sizeof(rcf_vf));
    wk->rf_s = rcf_alloc(RC_W * sizeof(rcf_vf));
    wk->cf = rcf_alloc(RC_H * sizeof(rcf_cvf));
    wk->cf_s = rcf_alloc(RC_H * sizeof(rcf_cvf));
    wk->q = rcf_alloc(2 * RC_SN * sizeof(float));
    wk->blk = rcf_alloc((RC_N / 25) * sizeof(float));
    wk->acc = rcf_alloc(RC_N * sizeof(uint64_t));
    wk->row = rcf_alloc(RC_W * sizeof(uint64_t));
    if (!wk->rd || !wk->rd_s || !wk->cd || !wk->cd_s || !wk->rf ||
        !wk->rf_s || !wk->cf || !wk->cf_s || !wk->q || !wk->blk ||
        !wk->acc || !wk->row) {
        rc_work_destroy(wk);
        return NULL;
    }
    return wk;
}

void rc_work_destroy(struct rc_work *wk)
{
    if (!wk)
        return;
    rcf_free(wk->rd);
    rcf_free(wk->rd_s);
    rcf_free(wk->cd);
    rcf_free(wk->cd_s);
    rcf_free(wk->rf);
    rcf_free(wk->rf_s);
    rcf_free(wk->cf);
    rcf_free(wk->cf_s);
    rcf_free(wk->q);
    rcf_free(wk->blk);
    rcf_free(wk->acc);
    rcf_free(wk->row);
    free(wk);
}

struct rc_frame *rc_frame_create(void)
{
    struct rc_frame *fr = calloc(1, sizeof(*fr));
    if (!fr)
        return NULL;
    fr->plane = rcf_alloc(RC_N * sizeof(float));
    fr->spec = rcf_alloc(2 * RC_SN * sizeof(float));
    if (!fr->plane || !fr->spec) {
        rc_frame_destroy(fr);
        return NULL;
    }
    return fr;
}

void rc_frame_destroy(struct rc_frame *fr)
{
    if (!fr)
        return;
    rcf_free(fr->plane);
    rcf_free(fr->spec);
    free(fr);
}

const float *rc_frame_plane(const struct rc_frame *fr)
{
    return fr->plane;
}

const float *rc_frame_spectrum_data(const struct rc_frame *fr)
{
    return fr->spec;
}

bool rc_frame_has_spectrum(const struct rc_frame *fr)
{
    return fr->has_spec;
}

// ---------------------------------------------------------------------------
// Plane (area_960)
// ---------------------------------------------------------------------------

static inline unsigned sample(const uint8_t *line, int i, int bps)
{
    if (bps == 1)
        return line[i];
    uint16_t v;
    memcpy(&v, line + 2 * (ptrdiff_t)i, sizeof(v));
    return v;
}

// Integer multiples: area_960 exactly (integer block sums, then float32 sum
// times float32(scale / (ky * kx))).
static void plane_multiple(float *plane, const uint8_t *src, ptrdiff_t stride,
                           int kx, int ky, int bps, double scale,
                           uint64_t *rowacc)
{
    const float f = (float)(scale / (double)(ky * kx));
    for (int Y = 0; Y < RC_H; Y++) {
        for (int X = 0; X < RC_W; X++)
            rowacc[X] = 0;
        for (int oy = 0; oy < ky; oy++) {
            const uint8_t *line = src + (ptrdiff_t)(Y * ky + oy) * stride;
            if (bps == 1) {
                for (int X = 0; X < RC_W; X++) {
                    unsigned s = 0;
                    for (int ox = 0; ox < kx; ox++)
                        s += line[X * kx + ox];
                    rowacc[X] += s;
                }
            } else {
                for (int X = 0; X < RC_W; X++) {
                    unsigned s = 0;
                    for (int ox = 0; ox < kx; ox++)
                        s += sample(line, X * kx + ox, 2);
                    rowacc[X] += s;
                }
            }
        }
        float *out = plane + (ptrdiff_t)Y * RC_W;
        for (int X = 0; X < RC_W; X++)
            out[X] = (float)rowacc[X] * f;
    }
}

// Other sizes: exact fractional area weights, like rt_small_sums. Source
// column i spans [i * RC_W, (i + 1) * RC_W) in units where output column X
// spans [X * w, (X + 1) * w) (rows likewise), so every output value's
// weights add up to w * h.
static void plane_area(float *plane, const uint8_t *src, ptrdiff_t stride,
                       int w, int h, int bps, double scale, uint64_t *acc,
                       uint64_t *row)
{
    memset(acc, 0, RC_N * sizeof(uint64_t));
    int Y = 0;
    for (int j = 0; j < h; j++) {
        const uint8_t *line = src + (ptrdiff_t)j * stride;
        memset(row, 0, RC_W * sizeof(uint64_t));
        int X = 0;
        for (int i = 0; i < w; i++) {
            uint64_t v = sample(line, i, bps);
            int64_t lo = (int64_t)i * RC_W, hi = lo + RC_W;
            while (lo < hi) {
                int64_t xend = (int64_t)(X + 1) * w;
                int64_t seg = hi < xend ? hi : xend;
                row[X] += (uint64_t)(seg - lo) * v;
                lo = seg;
                if (lo == xend && X + 1 < RC_W)
                    X++;
            }
        }
        int64_t lo = (int64_t)j * RC_H, hi = lo + RC_H;
        while (lo < hi) {
            int64_t yend = (int64_t)(Y + 1) * h;
            int64_t seg = hi < yend ? hi : yend;
            uint64_t wy = (uint64_t)(seg - lo);
            uint64_t *out = acc + (ptrdiff_t)Y * RC_W;
            for (int x = 0; x < RC_W; x++)
                out[x] += wy * row[x];
            lo = seg;
            if (lo == yend && Y + 1 < RC_H)
                Y++;
        }
    }
    const double f = scale / ((double)w * (double)h);
    for (int i = 0; i < RC_N; i++)
        plane[i] = (float)((double)acc[i] * f);
}

bool rc_frame_set_luma(struct rc_frame *fr, const void *luma, ptrdiff_t stride,
                       int w, int h, int bytes_per_sample, double scale,
                       struct rc_work *wk)
{
    fr->has_plane = fr->has_spec = false;
    if (!luma || w < 1 || h < 1 || w > (1 << 16) || h > (1 << 16) ||
        (bytes_per_sample != 1 && bytes_per_sample != 2) ||
        stride < (ptrdiff_t)w * bytes_per_sample)
        return false;
    if (w % RC_W == 0 && h % RC_H == 0) {
        plane_multiple(fr->plane, luma, stride, w / RC_W, h / RC_H,
                       bytes_per_sample, scale, wk->row);
    } else {
        plane_area(fr->plane, luma, stride, w, h, bytes_per_sample, scale,
                   wk->acc, wk->row);
    }
    fr->has_plane = true;
    return true;
}

// ---------------------------------------------------------------------------
// Spectrum: np.fft.rfft2((plane - plane.mean()) * win)
// ---------------------------------------------------------------------------

void rc_frame_spectrum(const struct rc_ctx *ctx, struct rc_frame *fr,
                       struct rc_work *wk)
{
    if (fr->has_spec || !fr->has_plane)
        return;
    const float mean = np_mean(fr->plane, RC_N);
    float *spec = fr->spec;

    // rfft along rows, in double (float32 input), stored as complex64
    double *rd = (double *)wk->rd;
    for (int y0 = 0; y0 < RC_H; y0 += RCF_VLD) {
        for (int l = 0; l < RCF_VLD; l++) {
            const int y = y0 + l;
            if (y >= RC_H) {
                for (int x = 0; x < RC_W; x++)
                    rd[x * RCF_VLD + l] = 0.0;
                continue;
            }
            const float *p = fr->plane + (ptrdiff_t)y * RC_W;
            const float *w = ctx->win + (ptrdiff_t)y * RC_W;
            for (int x = 0; x < RC_W; x++)
                rd[x * RCF_VLD + l] = (double)((p[x] - mean) * w[x]);
        }
        rcf_rfft_d(&ctx->rfd, wk->rd, wk->rd_s, true, 1.0);
        for (int l = 0; l < RCF_VLD && y0 + l < RC_H; l++) {
            float *o = spec + 2 * (ptrdiff_t)(y0 + l) * RC_FW;
            o[0] = (float)rd[l];
            o[1] = 0.0f;
            int i = 1, ii = 1;
            for (; i < RC_W - 1; i += 2, ii++) {
                o[2 * ii] = (float)rd[i * RCF_VLD + l];
                o[2 * ii + 1] = (float)rd[(i + 1) * RCF_VLD + l];
            }
            o[2 * ii] = (float)rd[i * RCF_VLD + l];  // i == RC_W - 1
            o[2 * ii + 1] = 0.0f;
        }
    }

    // fft along columns, in double (complex64 input), stored as complex64
    for (int c0 = 0; c0 < RC_FW; c0 += RCF_VLD) {
        for (int y = 0; y < RC_H; y++) {
            double *re = (double *)&wk->cd[y].r, *im = (double *)&wk->cd[y].i;
            const float *s = spec + 2 * ((ptrdiff_t)y * RC_FW + c0);
            for (int l = 0; l < RCF_VLD; l++) {
                const bool in = c0 + l < RC_FW;
                re[l] = in ? (double)s[2 * l] : 0.0;
                im[l] = in ? (double)s[2 * l + 1] : 0.0;
            }
        }
        rcf_cfft_d(&ctx->cfd, wk->cd, wk->cd_s, true, 1.0);
        for (int y = 0; y < RC_H; y++) {
            const double *re = (const double *)&wk->cd[y].r;
            const double *im = (const double *)&wk->cd[y].i;
            float *s = spec + 2 * ((ptrdiff_t)y * RC_FW + c0);
            for (int l = 0; l < RCF_VLD && c0 + l < RC_FW; l++) {
                s[2 * l] = (float)re[l];
                s[2 * l + 1] = (float)im[l];
            }
        }
    }
    fr->has_spec = true;
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

struct peak {
    float best, second;
    size_t idx;
    bool nan;
};

// shift_from_spectra up to the argmax: irfft2(r / (|r| + 1e-6), s=(540, 960))
// in float32, scanned in row-major order for numpy's argmax (first maximum;
// the first NaN wins) and the runner-up value.
static struct peak correlate(const struct rc_ctx *ctx, const float *fa,
                             const float *fb, struct rc_work *wk)
{
    // ifft along columns (fct float32(1/540))
    const float fct_h = 1.0f / (float)RC_H, fct_w = 1.0f / (float)RC_W;
    for (int c0 = 0; c0 < RC_FW; c0 += RCF_VLF) {
        for (int y = 0; y < RC_H; y++) {
            float *re = (float *)&wk->cf[y].r, *im = (float *)&wk->cf[y].i;
            const ptrdiff_t o = 2 * ((ptrdiff_t)y * RC_FW + c0);
            for (int l = 0; l < RCF_VLF; l++) {
                if (c0 + l < RC_FW) {
                    norm_cross(fa[o + 2 * l], fa[o + 2 * l + 1], fb[o + 2 * l],
                               fb[o + 2 * l + 1], &re[l], &im[l]);
                } else {
                    re[l] = im[l] = 0.0f;
                }
            }
        }
        rcf_cfft_f(&ctx->cff, wk->cf, wk->cf_s, false, fct_h);
        for (int y = 0; y < RC_H; y++) {
            const float *re = (const float *)&wk->cf[y].r;
            const float *im = (const float *)&wk->cf[y].i;
            float *q = wk->q + 2 * ((ptrdiff_t)y * RC_FW + c0);
            for (int l = 0; l < RCF_VLF && c0 + l < RC_FW; l++) {
                q[2 * l] = re[l];
                q[2 * l + 1] = im[l];
            }
        }
    }

    // irfft along rows (fct float32(1/960)): numpy's c2r input packing drops
    // the imaginary parts of bins 0 and 480
    struct peak pk = {-INFINITY, -INFINITY, 0, false};
    bool first = true;
    float *rf = (float *)wk->rf;
    for (int y0 = 0; y0 < RC_H; y0 += RCF_VLF) {
        for (int l = 0; l < RCF_VLF; l++) {
            const int y = y0 + l;
            if (y >= RC_H) {
                for (int x = 0; x < RC_W; x++)
                    rf[x * RCF_VLF + l] = 0.0f;
                continue;
            }
            const float *q = wk->q + 2 * (ptrdiff_t)y * RC_FW;
            rf[l] = q[0];
            int i = 1, ii = 1;
            for (; i < RC_W - 1; i += 2, ii++) {
                rf[i * RCF_VLF + l] = q[2 * ii];
                rf[(i + 1) * RCF_VLF + l] = q[2 * ii + 1];
            }
            rf[i * RCF_VLF + l] = q[2 * ii];  // i == RC_W - 1
        }
        rcf_rfft_f(&ctx->rff, wk->rf, wk->rf_s, false, fct_w);
        for (int l = 0; l < RCF_VLF && y0 + l < RC_H; l++) {
            const size_t base = (size_t)(y0 + l) * RC_W;
            for (int x = 0; x < RC_W; x++) {
                const float v = rf[x * RCF_VLF + l];
                if (v != v) {  // NaN: numpy's argmax returns the first one
                    if (!pk.nan)
                        pk = (struct peak){v, v, base + x, true};
                    continue;
                }
                if (pk.nan)
                    continue;
                if (first) {
                    pk.best = v;
                    pk.idx = base + x;
                    first = false;
                } else if (v > pk.best) {
                    pk.second = pk.best;
                    pk.best = v;
                    pk.idx = base + x;
                } else if (v > pk.second) {
                    pk.second = v;
                }
            }
        }
    }
    return pk;
}

// residual_after_shift(a, b, dy, dx) as float32 numpy computes it.
static double residual(const float *a, const float *b, int dy, int dx,
                       float *blk)
{
    const int pdy = dy > 0 ? dy : 0, ndy = dy < 0 ? -dy : 0;
    const int pdx = dx > 0 ? dx : 0, ndx = dx < 0 ? -dx : 0;
    // pa = a[ndy:h-pdy, ndx:w-pdx], pb = b[pdy:h-ndy, pdx:w-ndx]
    const int hh = (RC_H - pdy - ndy) / 5 * 5, ww = (RC_W - pdx - ndx) / 5 * 5;
    const int hb = hh / 5, wb = ww / 5;
    const float *pa = a + (ptrdiff_t)ndy * RC_W + ndx;
    const float *pb = b + (ptrdiff_t)pdy * RC_W + pdx;
    for (int Y = 0; Y < hb; Y++) {
        for (int X = 0; X < wb; X++) {
            // .reshape(hh//5, 5, ww//5, 5).mean((1, 3)): each block row
            // summed left to right (from -0.0), the rows added top to bottom
            // (from 0.0), then divided by 25
            float sa = 0.0f, sb = 0.0f;
            for (int j = 0; j < 5; j++) {
                const float *ra = pa + (ptrdiff_t)(5 * Y + j) * RC_W + 5 * X;
                const float *rb = pb + (ptrdiff_t)(5 * Y + j) * RC_W + 5 * X;
                float ta = -0.0f, tb = -0.0f;
                for (int k = 0; k < 5; k++) {
                    ta += ra[k];
                    tb += rb[k];
                }
                sa += ta;
                sb += tb;
            }
            sa = (float)((double)sa / 25.0);
            sb = (float)((double)sb / 25.0);
            blk[Y * wb + X] = fabsf(sa - sb);
        }
    }
    // abs(sa - sb).mean()
    return (double)np_mean(blk, (size_t)hb * wb);
}

struct rc_trans rc_transition(const struct rc_ctx *ctx, struct rc_frame *prev,
                              struct rc_frame *cur, double mad, double hold,
                              struct rc_work *wk, struct rc_diag *diag)
{
    struct rc_trans tr = {0};
    struct rc_diag dg = {0};
    dg.gated = rc_gated(mad, hold);
    if (!dg.gated || !prev->has_plane || !cur->has_plane)
        goto done;
    rc_frame_spectrum(ctx, prev, wk);
    rc_frame_spectrum(ctx, cur, wk);
    struct peak pk = correlate(ctx, prev->spec, cur->spec, wk);
    dg.correlated = true;
    dg.peak = pk.best;
    dg.second = pk.second;
    dg.peak_y = (int)(pk.idx / RC_W);
    dg.peak_x = (int)(pk.idx % RC_W);
    int dy = dg.peak_y < RC_H / 2 ? dg.peak_y : dg.peak_y - RC_H;
    int dx = dg.peak_x < RC_W / 2 ? dg.peak_x : dg.peak_x - RC_W;
    dy = -dy;
    dx = -dx;
    // implausible: a quarter of the frame or more
    if (abs(dy) >= RC_H / 4 || abs(dx) >= RC_W / 4)
        goto done;
    tr.has_shift = true;
    tr.dy = dy;
    tr.dx = dx;
    if (dy == 0 && dx == 0)
        goto done;
    dg.has_residual = true;
    dg.residual = residual(prev->plane, cur->plane, dy, dx, wk->blk);
    tr.camera_only = dg.residual < hold;
done:
    if (diag)
        *diag = dg;
    return tr;
}
