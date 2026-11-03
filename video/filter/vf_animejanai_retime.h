/*
 * Held-frame re-timing for vf_animejanai's RIFE (rife-retime=yes).
 *
 * Anime holds each drawing for 2-4 frames ("on twos/threes"). Plain RIFE
 * interpolates every consecutive source pair, so a held pair yields copies
 * of the same drawing and the motion arrives in bursts. Re-timing skips the
 * held frames and spreads RIFE across the whole hold, from one drawing to
 * the next, while the output stays factor x source frames on the same grid.
 *
 * Camera mode (rt_params.camera) also covers a pan or shake on ones over
 * characters on twos/threes: every frame changes, so the hold test never
 * fires and the characters keep juddering. A frame that is held once the
 * global shift is undone is "camera-only". Inside a span of at most maxcam
 * frames between two real drawings whose camera path is close to linear,
 * it counts as held, so RIFE moves camera and characters together. Shakes
 * (direction reversals), jerks and bent paths keep every frame. The known
 * cost: on slow pans on ones RIFE eases into each drawing across the span,
 * so the background pauses briefly there.
 *
 * This is a port of Planner in upscaler-pipeline's
 * services/worker/vpy/anime_retime.py at commit d28fca0, in both its
 * "retime" and its "camera" mode. Camera is the worker's default
 * (DEFAULT_MODE = "camera"), and VS_RETIME_MAXCAM sets its maxcam there.
 * Deliberate differences:
 *  - no CUT label and no cut window: the RIFE scene detector (aji's SCD,
 *    queried per morph pair by the filter) is the cut test, in place of the
 *    worker's local MAD cut rule and vsrife's scene-change test. The planner
 *    hands the filter the morph pair; on a scene change every slot of the
 *    position shows the source frame (the worker's "hold, then snap"). So
 *    the labels are HOLD, CAMERA and DRAWING, and only the guard refuses
 *    the drawing that closes a camera span;
 *  - a shift with a component beyond RT_SHIFT_MAX counts as not measured
 *    (the worker's metric reports none beyond a quarter frame).
 * The guard (no morph over 2+ source frames across a MAD above 22) is kept.
 * The camera path test decides in exact integer arithmetic. For shifts up to
 * RT_SHIFT_MAX that provably gives the worker's float answer, epsilons
 * included (see the parity harness README).
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
#define RT_MAXCAM_DEFAULT 4     // longest merged camera span (VS_RETIME_MAXCAM)
#define RT_MAXCAM_MAX 8
#define RT_FACTOR_MAX 8
// Largest |dy| and |dx| (960x540 px) the camera path test takes. A 960x540
// phase correlation cannot report more.
#define RT_SHIFT_MAX 1023

// Stream length when the end of the stream is not known yet.
#define RT_N_UNKNOWN INT64_MAX

struct rt_params {
    int factor;      // output frames per source frame, 2..RT_FACTOR_MAX
    int maxhold;     // 1..RT_MAXHOLD_MAX
    double hold;     // RT_HOLD_DEFAULT
    double guard;    // RT_GUARD
    bool camera;     // camera mode (the worker's default)
    int maxcam;      // 1..RT_MAXCAM_MAX; rt_maxcam() is what applies
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

// Metrics of the transition from frame i - 1 to frame i: Python's
// Transition without sc_mad (the scene test is the filter's). Plain mode
// reads only mad.
struct rt_trans {
    double mad;        // hold metric
    bool has_shift;    // a global shift was measured (Python: shift not None)
    int dy, dx;        // that shift d, frame i ~ frame i - 1 moved by d, in
                       // 960x540 px
    bool camera_only;  // frame i is held once the shift is undone
};

// get(ctx, i): the transition into frame i, 1 <= i < n.
typedef struct rt_trans (*rt_trans_fn)(void *ctx, int64_t i);

// The longest camera span that may be merged: Python's
// max(2, min(maxcam, maxhold)), 0 in plain mode. A merged span thus always
// re-times as a short hold. (The RT_MAXCAM_MAX cap only matters for
// out-of-range parameters.)
static inline int rt_maxcam(const struct rt_params *rp)
{
    if (!rp->camera)
        return 0;
    int m = rp->maxcam < rp->maxhold ? rp->maxcam : rp->maxhold;
    if (m > RT_MAXCAM_MAX)
        m = RT_MAXCAM_MAX;
    return m > 2 ? m : 2;
}

// Windows around source position p.
//  - Frames rt_plan(p) and rt_slot() name (morph endpoints and p itself):
//    p - rt_end_behind() .. p + rt_end_ahead().
//  - Transitions rt_plan(p) reads: p - rt_behind() .. p + rt_ahead(), so the
//    metric needs frames p - rt_behind() - 1 .. p + rt_ahead().
// In plain mode the two are the same. Camera mode widens the read window
// behind p by rt_maxcam() - 1: a candidate key as far back as
// p - maxhold + 1 looks that much further back for the drawing before it.
// Ahead it reads no further than max(maxhold, rt_maxcam()) (maxhold unless
// maxhold is 1): a candidate key after p is only bridged from the last key
// at or before p, so its closing drawing lies within rt_maxcam() of p. The
// worker's lookahead also adds maxcam - 1 ahead; its planner never reads
// that far either.
static inline int rt_end_behind(const struct rt_params *rp)
{
    return rp->maxhold - 1;
}

static inline int rt_end_ahead(const struct rt_params *rp)
{
    return rp->maxhold;
}

static inline int rt_behind(const struct rt_params *rp)
{
    return rt_end_behind(rp) + (rp->camera ? rt_maxcam(rp) - 1 : 0);
}

static inline int rt_ahead(const struct rt_params *rp)
{
    const int mc = rt_maxcam(rp);
    return mc > rp->maxhold ? mc : rp->maxhold;
}

// The largest rt_behind() and rt_ahead() over valid parameters.
#define RT_BEHIND_MAX (RT_MAXHOLD_MAX - 1 + RT_MAXCAM_MAX - 1)
#define RT_AHEAD_MAX RT_MAXHOLD_MAX

enum rt_label {
    RT_LABEL_HOLD,     // same drawing as the frame before
    RT_LABEL_CAMERA,   // camera mode: held once the camera shift is undone
    RT_LABEL_DRAWING,  // a new drawing
};

// The hold test is Python's exact one (mad < hold is held), so a NaN metric
// counts as a new drawing there too.
static inline enum rt_label rt_label(const struct rt_params *rp,
                                     const struct rt_trans *tr)
{
    if (tr->mad < rp->hold)
        return RT_LABEL_HOLD;
    if (rp->camera && tr->camera_only)
        return RT_LABEL_CAMERA;
    return RT_LABEL_DRAWING;
}

// Frame 0 or a new drawing.
static inline bool rt_is_real_key(const struct rt_params *rp, int64_t j,
                                  rt_trans_fn get, void *ctx)
{
    if (j == 0)
        return true;
    const struct rt_trans tr = get(ctx, j);
    return rt_label(rp, &tr) == RT_LABEL_DRAWING;
}

// Python's camera_path_is_smooth() on the n steps (dy[i], dx[i]) of a span,
// (0, 0) for a held frame; 0 <= n <= RT_MAXCAM_MAX, |dy|, |dx| <=
// RT_SHIFT_MAX. Would a linear camera path from the first to the last frame
// stay close to the measured one? Rejects direction reversals (shakes),
// jerks (moving steps more than 3x apart in size) and paths that stray more
// than max(1, 0.5 * the largest step) px from the straight line (the step
// term admits a camera that moves on twos).
//
// Python compares float square roots with a 1e-9 slack (its x ** 0.5 is libm
// pow, not always the correctly rounded sqrt, so a float port would only
// reproduce its intermediates with the same libm). Here the same tests run
// on exact squares: size(big) > 3 * size(small) + 1e-9 becomes
// big2 > 9 * small2, and |dev_m| > max(1, size(big) / 2) + 1e-9, where
// dev_m = (n * c_m - m * t) / n (c_m: sum of the first m steps, t: sum of
// all), becomes 4 * |n * c_m - m * t|^2 > max(4, big2) * n^2. In these
// bounds an exact tie sits deep inside Python's slack and any other value
// at least 2e-6 beyond it, while Python's rounding errors stay below 1e-11,
// so both always agree.
static inline bool rt_camera_path_is_smooth(int n, const int *dy,
                                            const int *dx)
{
    int64_t big2 = 0, small2 = 0;  // squared sizes of the moving steps
    bool moving = false;
    for (int i = 0; i < n; i++) {
        if (!dy[i] && !dx[i])
            continue;
        for (int k = i + 1; k < n; k++) {
            // a direction reversal (a held step's product is 0)
            if ((int64_t)dy[i] * dy[k] + (int64_t)dx[i] * dx[k] < 0)
                return false;
        }
        const int64_t s2 = (int64_t)dy[i] * dy[i] + (int64_t)dx[i] * dx[i];
        if (!moving || s2 > big2)
            big2 = s2;
        if (!moving || s2 < small2)
            small2 = s2;
        moving = true;
    }
    if (!moving)
        return false;
    if (big2 > 9 * small2)
        return false;  // a jerk
    int64_t ty = 0, tx = 0;
    for (int i = 0; i < n; i++) {
        ty += dy[i];
        tx += dx[i];
    }
    const int64_t lim = (big2 > 4 ? big2 : 4) * n * n;
    int64_t cy = 0, cx = 0;
    for (int m = 1; m < n; m++) {
        cy += dy[m - 1];
        cx += dx[m - 1];
        const int64_t ey = n * cy - m * ty, ex = n * cx - m * tx;
        if (4 * (ey * ey + ex * ex) > lim)
            return false;  // strays from the straight line
    }
    return true;
}

// Is camera-only frame c (1 <= c < n) inside a bounded, near-linear camera
// span r0 < c < r1 between two real drawings, r1 - r0 <= rt_maxcam(), that
// RIFE may bridge? Python's _camera_merged(), whose CUT test on r1 has no
// counterpart here (see the top).
static inline bool rt_camera_merged(const struct rt_params *rp, int64_t c,
                                    int64_t n, rt_trans_fn get, void *ctx)
{
    const int64_t maxcam = rt_maxcam(rp);
    int64_t r0 = -1, r1 = -1;
    for (int64_t j = c - 1; j > c - maxcam && j >= 0; j--) {
        if (rt_is_real_key(rp, j, get, ctx)) {
            r0 = j;
            break;
        }
    }
    if (r0 < 0)
        return false;
    const int64_t end = r0 + maxcam < n - 1 ? r0 + maxcam : n - 1;
    for (int64_t j = c + 1; j <= end; j++) {
        if (rt_is_real_key(rp, j, get, ctx)) {
            r1 = j;
            break;
        }
    }
    if (r1 < 0 || get(ctx, r1).mad > rp->guard)
        return false;
    int dy[RT_MAXCAM_MAX], dx[RT_MAXCAM_MAX];
    int steps = 0;
    for (int64_t j = r0 + 1; j <= r1; j++) {
        const struct rt_trans tr = get(ctx, j);
        if (rt_label(rp, &tr) == RT_LABEL_HOLD) {
            dy[steps] = dx[steps] = 0;
        } else if (tr.has_shift && tr.dy >= -RT_SHIFT_MAX &&
                   tr.dy <= RT_SHIFT_MAX && tr.dx >= -RT_SHIFT_MAX &&
                   tr.dx <= RT_SHIFT_MAX) {
            dy[steps] = tr.dy;
            dx[steps] = tr.dx;
        } else {
            return false;  // no usable shift
        }
        steps++;
    }
    return rt_camera_path_is_smooth(steps, dy, dx);
}

// Is frame j (0 <= j < n) a key, a frame morphs start and end on? Frame 0,
// a new drawing, or a camera-only frame that no camera span absorbs.
static inline bool rt_is_key(const struct rt_params *rp, int64_t j, int64_t n,
                             rt_trans_fn get, void *ctx)
{
    if (j == 0)
        return true;
    const struct rt_trans tr = get(ctx, j);
    const enum rt_label lab = rt_label(rp, &tr);
    if (lab == RT_LABEL_CAMERA)
        return !rt_camera_merged(rp, j, n, get, ctx);
    return lab == RT_LABEL_DRAWING;
}

// Plan source position p (0 <= p < n) of a stream of n frames (or
// RT_N_UNKNOWN while at least rt_ahead() frames follow p). Calls get() only
// for 1 <= i <= min(p + rt_ahead(), n - 1), i >= p - rt_behind().
static inline struct rt_plan rt_plan(const struct rt_params *rp, int64_t p,
                                     int64_t n, rt_trans_fn get, void *ctx)
{
    const int64_t maxhold = rp->maxhold;
    int64_t a = -1, b = -1;
    for (int64_t j = p; j > p - maxhold && j >= 0; j--) {
        if (rt_is_key(rp, j, n, get, ctx)) {
            a = j;
            break;
        }
    }
    if (a >= 0) {
        int64_t end = a + maxhold < n - 1 ? a + maxhold : n - 1;
        for (int64_t j = p + 1; j <= end; j++) {
            if (rt_is_key(rp, j, n, get, ctx)) {
                b = j;
                break;
            }
        }
    }
    if (a >= 0 && b >= 0) {
        // Short hold a..b-1 (b - a <= maxhold): spread RIFE across it. The
        // guard only applies where re-timing stretches a morph over 2+
        // source frames; a 1-frame step morphs exactly like plain RIFE.
        if (b - a >= 2 && get(ctx, b).mad > rp->guard)
            return (struct rt_plan){RT_PLAN_HOLD, p, -1};
        return (struct rt_plan){RT_PLAN_SPAN, a, b};
    }
    // Long hold or the stream's end: plain behaviour (1-frame steps, no
    // guard), minus RIFE on held pairs.
    int64_t nxt = p + 1;
    if (nxt >= n || get(ctx, nxt).mad < rp->hold)
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
