/*
 * Held-frame re-timing for vf_animejanai's RIFE (rife-retime=yes).
 *
 * Anime holds each drawing for 2-4 frames ("on twos/threes"). Plain RIFE
 * interpolates every consecutive source pair, so a held pair yields copies
 * of the same drawing and the motion arrives in bursts. Re-timing skips the
 * held frames and spreads RIFE across the whole hold, from one drawing to
 * the next, while the output stays factor x source frames on the same grid.
 *
 * This is a port of the "retime" mode of Planner in upscaler-pipeline's
 * services/worker/vpy/anime_retime.py (commit ecdb83d), with two
 * deliberate differences:
 *  - no camera mode (it is opt-in there, and needs phase correlation);
 *  - no cut window: the RIFE scene detector (aji's SCD, queried per
 *    morph pair by the filter) is the cut test, in place of the worker's
 *    local MAD cut rule and vsrife's scene-change test. The planner hands
 *    the filter the morph pair; on a scene change every slot of the
 *    position shows the source frame (the worker's "hold, then snap").
 * The guard (no morph over 2+ source frames across a MAD above 22) is kept.
 *
 * Self-contained C99 with no mpv dependencies, so it compiles and tests on
 * any host (see the parity harness that checks it against the Python
 * Planner).
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

#ifndef MP_VF_ANIMEJANAI_RETIME_H
#define MP_VF_ANIMEJANAI_RETIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Hold metric: mean absolute difference of 192x108 area-averaged luma, in
// full-range 8-bit levels (the worker's and the 2026-10-31 bake-off's
// metric: ffmpeg scale=192:108:flags=area,format=gray).
#define RT_SMALL_W 192
#define RT_SMALL_H 108
#define RT_SMALL_N (RT_SMALL_W * RT_SMALL_H)

#define RT_HOLD_DEFAULT 0.6     // MAD below this: same drawing as before
#define RT_GUARD 22.0           // never stretch a morph across a bigger jump
#define RT_MAXHOLD_DEFAULT 4    // longer holds are deliberate pauses
#define RT_MAXHOLD_MAX 8
#define RT_FACTOR_MAX 8

// Stream length when the end of the stream is not known yet.
#define RT_N_UNKNOWN INT64_MAX

struct rt_params {
    int factor;      // output frames per source frame, 2..RT_FACTOR_MAX
    int maxhold;     // 1..RT_MAXHOLD_MAX
    double hold;     // RT_HOLD_DEFAULT
    double guard;    // RT_GUARD
};

enum rt_plan_kind {
    RT_PLAN_HOLD,    // every slot shows source frame p
    RT_PLAN_SPAN,    // morph from key a to key b across the hold a..b-1
    RT_PLAN_STEP,    // a long hold ends: plain morph from p to b = p + 1
};

struct rt_plan {
    enum rt_plan_kind kind;
    int64_t a, b;    // SPAN/STEP: the morph pair (scene-test it first)
};

// One output slot: show source frame a, or RIFE(a, b, t) with 0 < t < 1.
struct rt_step {
    bool interp;
    int64_t a, b;
    double t;
};

// mad(ctx, i): hold metric of the transition from frame i - 1 to frame i,
// 1 <= i < n.
typedef double (*rt_mad_fn)(void *ctx, int64_t i);

// Source frames rt_plan(p) may name as a morph endpoint: p - behind ..
// p + ahead. The metric reads transitions p - behind .. p + ahead (and
// thus needs frame p - behind - 1 for the first one).
static inline int rt_behind(const struct rt_params *rp) { return rp->maxhold - 1; }
static inline int rt_ahead(const struct rt_params *rp) { return rp->maxhold; }

static inline bool rt_is_key(const struct rt_params *rp, int64_t j,
                             rt_mad_fn mad, void *ctx)
{
    return j == 0 || mad(ctx, j) >= rp->hold;
}

// Plan source position p (0 <= p < n) of a stream of n frames (or
// RT_N_UNKNOWN while more than rt_ahead() frames follow p). Calls mad()
// only for 1 <= i <= min(p + rt_ahead(), n - 1), i >= p - rt_behind().
static inline struct rt_plan rt_plan(const struct rt_params *rp, int64_t p,
                                     int64_t n, rt_mad_fn mad, void *ctx)
{
    const int64_t maxhold = rp->maxhold;
    int64_t a = -1, b = -1;
    for (int64_t j = p; j > p - maxhold && j >= 0; j--) {
        if (rt_is_key(rp, j, mad, ctx)) {
            a = j;
            break;
        }
    }
    if (a >= 0) {
        int64_t end = a + maxhold < n - 1 ? a + maxhold : n - 1;
        for (int64_t j = p + 1; j <= end; j++) {
            if (rt_is_key(rp, j, mad, ctx)) {
                b = j;
                break;
            }
        }
    }
    if (a >= 0 && b >= 0) {
        // Short hold a..b-1 (b - a <= maxhold): spread RIFE across it. The
        // guard only applies where re-timing stretches a morph over 2+
        // source frames; a 1-frame step morphs exactly like plain RIFE.
        if (b - a >= 2 && mad(ctx, b) > rp->guard)
            return (struct rt_plan){RT_PLAN_HOLD, p, -1};
        return (struct rt_plan){RT_PLAN_SPAN, a, b};
    }
    // Long hold or the stream's end: plain behaviour (1-frame steps, no
    // guard), minus RIFE on held pairs.
    int64_t nxt = p + 1;
    if (nxt >= n || mad(ctx, nxt) < rp->hold)
        return (struct rt_plan){RT_PLAN_HOLD, p, -1};
    return (struct rt_plan){RT_PLAN_STEP, p, nxt};
}

// Output slot k (0 <= k < factor) of position p under plan pl. scene: the
// scene test fired on the plan's pair (a, b); ignored for RT_PLAN_HOLD.
static inline struct rt_step rt_slot(const struct rt_params *rp,
                                     const struct rt_plan *pl, int64_t p,
                                     int k, bool scene)
{
    const struct rt_step src = {false, p, -1, 0.0};
    if (pl->kind == RT_PLAN_HOLD || scene)
        return src;
    if (pl->kind == RT_PLAN_SPAN) {
        int64_t span = (int64_t)rp->factor * (pl->b - pl->a);
        int64_t j = (p - pl->a) * rp->factor + k;
        if (j == 0)
            return src;  // p == a
        return (struct rt_step){true, pl->a, pl->b, (double)j / (double)span};
    }
    if (k == 0)
        return src;
    return (struct rt_step){true, p, pl->b, (double)k / (double)rp->factor};
}

// ---------------------------------------------------------------------------
// Hold metric. A frame's 192x108 area average is kept as exact integer sums:
// sums[y * 192 + x] = sum of sample * overlap over the output pixel's source
// footprint, with overlaps in units where one source pixel spans 192 x 108,
// so every output pixel's weights add up to w * h. Integer sizes multiple of
// 192x108 reduce to plain block sums (x 192 * 108); other sizes get exact
// fractional area weights (ffmpeg's area scaler).
// ---------------------------------------------------------------------------

// Factor from raw luma code values (in a container of `bits` bits, sample
// in the high bits for P010) to full-range 8-bit levels. Offsets cancel in
// a MAD. Limited range expands by 255/219, like the bake-off's grey convert.
static inline double rt_luma_scale(int bits, bool limited)
{
    return (limited ? 255.0 / 219.0 : 1.0) / (double)(1 << (bits - 8));
}

// plane: luma samples (bytes_per_sample 1 = 8-bit, 2 = 16-bit containers),
// stride in bytes. row: scratch for RT_SMALL_W values. sums: RT_SMALL_N out.
static inline void rt_small_sums(const void *plane, ptrdiff_t stride, int w,
                                 int h, int bytes_per_sample, uint64_t *row,
                                 uint64_t *sums)
{
    for (int i = 0; i < RT_SMALL_N; i++)
        sums[i] = 0;
    const bool even_x = w % RT_SMALL_W == 0;
    const int bw = w / RT_SMALL_W;
    int Y = 0;
    for (int j = 0; j < h; j++) {
        const uint8_t *line = (const uint8_t *)plane + (ptrdiff_t)j * stride;
        for (int X = 0; X < RT_SMALL_W; X++)
            row[X] = 0;
        if (even_x) {
            // each source pixel lies in one output column, overlap 192
            for (int X = 0, i = 0; X < RT_SMALL_W; X++) {
                uint64_t s = 0;
                if (bytes_per_sample == 1) {
                    for (int e = i + bw; i < e; i++)
                        s += line[i];
                } else {
                    const uint16_t *l16 = (const uint16_t *)line;
                    for (int e = i + bw; i < e; i++)
                        s += l16[i];
                }
                row[X] = s * RT_SMALL_W;
            }
        } else {
            int X = 0;
            for (int i = 0; i < w; i++) {
                uint64_t v = bytes_per_sample == 1
                    ? line[i] : ((const uint16_t *)line)[i];
                int64_t lo = (int64_t)i * RT_SMALL_W, hi = lo + RT_SMALL_W;
                while (lo < hi) {
                    int64_t xend = (int64_t)(X + 1) * w;
                    int64_t seg = hi < xend ? hi : xend;
                    row[X] += (uint64_t)(seg - lo) * v;
                    lo = seg;
                    if (lo == xend && X + 1 < RT_SMALL_W)
                        X++;
                }
            }
        }
        // vertical: source row j spans [j * 108, j * 108 + 108), output
        // row Y spans [Y * h, Y * h + h)
        int64_t lo = (int64_t)j * RT_SMALL_H, hi = lo + RT_SMALL_H;
        while (lo < hi) {
            int64_t yend = (int64_t)(Y + 1) * h;
            int64_t seg = hi < yend ? hi : yend;
            uint64_t wy = (uint64_t)(seg - lo);
            uint64_t *out = sums + (ptrdiff_t)Y * RT_SMALL_W;
            for (int X = 0; X < RT_SMALL_W; X++)
                out[X] += wy * row[X];
            lo = seg;
            if (lo == yend && Y + 1 < RT_SMALL_H)
                Y++;
        }
    }
}

// MAD between two frames' sums (same w x h), in full-range 8-bit levels.
static inline double rt_small_mad(const uint64_t *sa, const uint64_t *sb,
                                  int w, int h, double scale)
{
    uint64_t acc = 0;
    for (int i = 0; i < RT_SMALL_N; i++)
        acc += sa[i] > sb[i] ? sa[i] - sb[i] : sb[i] - sa[i];
    return (double)acc * scale / ((double)w * (double)h * (double)RT_SMALL_N);
}

#endif
