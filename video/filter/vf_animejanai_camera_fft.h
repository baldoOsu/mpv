/*
 * Real and complex FFTs for vf_animejanai's camera metric: a C port of the
 * FFTPACK-style kernels of pocketfft's C++ header (the copy numpy 2.x uses for
 * numpy.fft), limited to the factors 4, 2, 3 and 5, in float and in double.
 * Same operations in the same order as pocketfft, so with the same inputs it
 * returns the same bits as numpy.fft (see the camera_metric parity harness).
 *
 * Each transform runs on RCF_VLF (float) or RCF_VLD (double) independent
 * signals at once, one per vector lane, like pocketfft's own multi-transform
 * vectorization: lanes never mix, so every lane gets exactly the scalar
 * result.
 *
 * Ported from pocketfft_hdronly.h, https://github.com/mreineck/pocketfft
 * commit 33ae5dc94c9cdc7f1c78346504a85de87cadaa12 (C++ branch; the version
 * numpy 2.3.4 vendors), file sha256
 * 50f4ddbe60432325f560c706b6f1c49f9e0949fc7f20ae6499f2299e37d9790c.
 *
 * Copyright (C) 2010-2022 Max-Planck-Society
 * Copyright (C) 2019-2020 Peter Bell
 * Authors: Martin Reinecke, Peter Bell
 * C port for mpv's vf_animejanai (2026).
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MP_VF_ANIMEJANAI_CAMERA_FFT_H
#define MP_VF_ANIMEJANAI_CAMERA_FFT_H

#include <stdbool.h>
#include <stddef.h>

// Vectors of RCF_VLF floats / RCF_VLD doubles (16 bytes, 32 with AVX).
// may_alias: callers fill and read lanes through scalar pointers.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(RCF_NO_VECTORS)
#if defined(__AVX__)
#define RCF_VBYTES 32
#else
#define RCF_VBYTES 16
#endif
#define RCF_VLF (RCF_VBYTES / 4)
#define RCF_VLD (RCF_VBYTES / 8)
typedef float rcf_vf __attribute__((vector_size(RCF_VBYTES), __may_alias__));
typedef double rcf_vd __attribute__((vector_size(RCF_VBYTES), __may_alias__));
#else
#define RCF_VLF 1
#define RCF_VLD 1
typedef float rcf_vf;
typedef double rcf_vd;
#endif

#define RCF_ALIGN 64
#define RCF_MAXFCT 16

struct rcf_cf {  // twiddle factors
    float r, i;
};
struct rcf_cd {
    double r, i;
};

typedef struct {  // RCF_VLF complex values
    rcf_vf r, i;
} rcf_cvf;
typedef struct {  // RCF_VLD complex values
    rcf_vd r, i;
} rcf_cvd;

// Plans own their twiddle tables (malloc in init, free in uninit) and are
// read-only afterwards, so one plan can serve any number of threads.
struct rcf_rplan_f {
    size_t n, nfct;
    size_t fct[RCF_MAXFCT];
    const float *tw[RCF_MAXFCT];
    float *mem;
};
struct rcf_rplan_d {
    size_t n, nfct;
    size_t fct[RCF_MAXFCT];
    const double *tw[RCF_MAXFCT];
    double *mem;
};
struct rcf_cplan_f {
    size_t n, nfct;
    size_t fct[RCF_MAXFCT];
    const struct rcf_cf *tw[RCF_MAXFCT];
    struct rcf_cf *mem;
};
struct rcf_cplan_d {
    size_t n, nfct;
    size_t fct[RCF_MAXFCT];
    const struct rcf_cd *tw[RCF_MAXFCT];
    struct rcf_cd *mem;
};

// false if n needs a factor other than 4, 2, 3 or 5 (rfftp) / 4, 3 or 5
// (cfftp): pocketfft's radix-7/8/11, generic and Bluestein paths are not
// ported. Also false on allocation failure.
bool rcf_rplan_init_f(struct rcf_rplan_f *p, size_t n);
bool rcf_rplan_init_d(struct rcf_rplan_d *p, size_t n);
void rcf_rplan_uninit_f(struct rcf_rplan_f *p);
void rcf_rplan_uninit_d(struct rcf_rplan_d *p);
bool rcf_cplan_init_f(struct rcf_cplan_f *p, size_t n);
bool rcf_cplan_init_d(struct rcf_cplan_d *p, size_t n);
void rcf_cplan_uninit_f(struct rcf_cplan_f *p);
void rcf_cplan_uninit_d(struct rcf_cplan_d *p);

// pocketfft rfftp::exec: in place on c[0..n), FFTPACK half-complex order
// (r0, r1, i1, ..., [r_n/2]); ch: scratch of n vectors; result times fct.
void rcf_rfft_f(const struct rcf_rplan_f *p, rcf_vf *c, rcf_vf *ch,
                bool forward, float fct);
void rcf_rfft_d(const struct rcf_rplan_d *p, rcf_vd *c, rcf_vd *ch,
                bool forward, double fct);
// pocketfft cfftp::exec: in place on c[0..n); ch: scratch of n values.
void rcf_cfft_f(const struct rcf_cplan_f *p, rcf_cvf *c, rcf_cvf *ch,
                bool forward, float fct);
void rcf_cfft_d(const struct rcf_cplan_d *p, rcf_cvd *c, rcf_cvd *ch,
                bool forward, double fct);

// RCF_ALIGN-aligned allocation (vector loads need it); free with rcf_free.
void *rcf_alloc(size_t size);
void rcf_free(void *ptr);

#endif
