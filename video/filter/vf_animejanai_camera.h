/*
 * Camera-mode transition metric for vf_animejanai's re-timing: per source
 * transition i-1 -> i, the global shift (phase correlation of 960x540 grey
 * planes) and whether frame i is "camera-only", i.e. held once that shift is
 * undone (a pan or shake over held drawings).
 *
 * Port of area_960 / spectrum / shift_from_spectra / residual_after_shift /
 * transition_metrics in upscaler-pipeline's services/worker/vpy/
 * anime_retime.py (commit d28fca0), as numpy 2.3.4 (the worker image's)
 * computes them on x86-64 with AVX2+FMA3: for luma sizes that are integer
 * multiples of 960x540 the results are bit-identical to the worker's (see
 * the camera_metric parity harness). Other sizes: the worker bilinear-resizes
 * to 1920x1080 first (zimg); here they get exact fractional area weights
 * straight to 960x540, which keeps the residual consistent with the 192x108
 * hold MAD (rt_small_sums uses the same weights).
 *
 * Self-contained C with no mpv dependencies. FFTs: vf_animejanai_camera_fft.c
 * (BSD-3, ported from pocketfft, numpy's FFT).
 *
 * Use (the plane and spectrum are per frame, so each frame's are computed
 * once and reused by the transitions into and out of it):
 *
 *     ctx = rc_ctx_create();                      // once; read-only after
 *     wk  = rc_work_create();                     // one per thread
 *     fr[k] = rc_frame_create();                  // a few, recycled
 *     rc_frame_set_luma(fr[i], y, stride, w, h, bps, rt_luma_scale(...), wk);
 *     tr = rc_transition(ctx, fr[i-1], fr[i], mad, hold, wk, NULL);
 *
 * Threading: rc_ctx is immutable after creation and may be shared by any
 * number of threads. An rc_work is scratch for one call at a time. A frame
 * is written by rc_frame_set_luma and rc_frame_spectrum (rc_transition calls
 * the latter for a gated transition whose frames lack a spectrum), so a
 * frame must not be used by two calls at once unless rc_frame_spectrum has
 * already been run on it, after which rc_transition only reads it. Nothing
 * allocates after the create functions; no global state.
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

#ifndef MP_VF_ANIMEJANAI_CAMERA_H
#define MP_VF_ANIMEJANAI_CAMERA_H

#include <stdbool.h>
#include <stddef.h>

#define RC_W 960            // metric plane (anime_retime METRIC_W x METRIC_H)
#define RC_H 540
#define RC_CUT_MAD 18.0     // anime_retime CUT_MAD: no shift estimated above

struct rc_ctx;    // Hann window + FFT plans
struct rc_work;   // per-thread scratch (about 8 MB, 12 MB with non-multiple sizes)
struct rc_frame;  // one frame's 960x540 plane and spectrum (about 4 MB)

// Transition i-1 -> i (anime_retime.Transition minus mad / sc_mad).
// has_shift false <=> Transition.shift is None; then dy = dx = 0 and
// camera_only = false. Shift (dy, dx) is in 960x540 px with
// b(x) ~ a(x - d), as in shift_from_spectra.
struct rc_trans {
    bool has_shift;
    int dy, dx;
    bool camera_only;
};

// Optional per-transition details for logging and parity checks.
struct rc_diag {
    bool gated;           // hold <= mad <= RC_CUT_MAD
    bool correlated;      // the correlation surface was computed
    float peak, second;   // its maximum and the largest other value
    int peak_y, peak_x;   // argmax (first in row-major order), unwrapped
    bool has_residual;    // residual_after_shift was evaluated
    double residual;      // its value (compared with hold)
};

struct rc_ctx *rc_ctx_create(void);
void rc_ctx_destroy(struct rc_ctx *ctx);
struct rc_work *rc_work_create(void);
void rc_work_destroy(struct rc_work *wk);
struct rc_frame *rc_frame_create(void);
void rc_frame_destroy(struct rc_frame *fr);

// The worker's camera gate (transition_metrics): a shift is only estimated
// when hold <= mad <= RC_CUT_MAD. mad: the 192x108 hold MAD of the
// transition (rt_small_mad); NaN never gates.
static inline bool rc_gated(double mad, double hold)
{
    return hold <= mad && mad <= RC_CUT_MAD;
}

// Frame part 1: the 960x540 float plane of a luma plane (w x h samples of
// bytes_per_sample 1 or 2, stride in bytes), in full-range 8-bit levels
// (scale = rt_luma_scale(8 * bytes_per_sample, limited)). Invalidates the
// frame's spectrum. false (frame unusable) on bad arguments.
bool rc_frame_set_luma(struct rc_frame *fr, const void *luma, ptrdiff_t stride,
                       int w, int h, int bytes_per_sample, double scale,
                       struct rc_work *wk);

// Frame part 2: the spectrum (mean removed, Hann window, rfft2). No-op if
// present; requires a plane.
void rc_frame_spectrum(const struct rc_ctx *ctx, struct rc_frame *fr,
                       struct rc_work *wk);
bool rc_frame_has_spectrum(const struct rc_frame *fr);

// The per-transition part of transition_metrics(prev, cur, camera=True)
// with HOLD_MAD = hold for both the gate and the camera-only test. Computes
// missing spectra of gated transitions. diag may be NULL.
struct rc_trans rc_transition(const struct rc_ctx *ctx, struct rc_frame *prev,
                              struct rc_frame *cur, double mad, double hold,
                              struct rc_work *wk, struct rc_diag *diag);

// Read access for tests: the 960x540 plane (row-major) and the spectrum
// (540 x 481 complex as re/im float pairs, numpy's rfft2 layout).
const float *rc_frame_plane(const struct rc_frame *fr);
const float *rc_frame_spectrum_data(const struct rc_frame *fr);

#endif
