/*
 * Real and complex FFTs for vf_animejanai's camera metric: a C port of the
 * FFTPACK-style kernels of pocketfft's C++ header (radix 4, 3, 5; radix 2
 * for rfftp), see vf_animejanai_camera_fft.h for what and why. The kernels
 * are in vf_animejanai_camera_fft_tmpl.h, instantiated here for float and
 * double (pocketfft's templates on T0).
 *
 * Ported from pocketfft_hdronly.h, https://github.com/mreineck/pocketfft
 * commit 33ae5dc94c9cdc7f1c78346504a85de87cadaa12, file sha256
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

// numpy builds pocketfft without FMA (x86-64 baseline), so no contraction.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vf_animejanai_camera_fft.h"

void *rcf_alloc(size_t size)
{
    unsigned char *raw = malloc(size + RCF_ALIGN + sizeof(void *));
    if (!raw)
        return NULL;
    uintptr_t a = (uintptr_t)(raw + sizeof(void *));
    a = (a + RCF_ALIGN - 1) & ~(uintptr_t)(RCF_ALIGN - 1);
    unsigned char *res = (unsigned char *)a;
    memcpy(res - sizeof(void *), &raw, sizeof(void *));
    return res;
}

void rcf_free(void *ptr)
{
    if (!ptr)
        return;
    void *raw;
    memcpy(&raw, (unsigned char *)ptr - sizeof(void *), sizeof(void *));
    free(raw);
}

// ---------------------------------------------------------------------------
// Twiddle factors: pocketfft sincos_2pibyn<float> (computed in double).
//
// pocketfft takes them from the C library's cos/sin, so numpy's FFT depends
// on the libm it runs on. For the two lengths the camera metric uses they
// are tabulated below: the values of glibc 2.35 (this port's reference host)
// and 2.42 (the worker image), all 178 correctly rounded (checked with
// mpmath). mingw-w64's libm misrounds one of them (n = 540, v1[31].r, by one
// ulp), which changed one spectrum value in about 0.4% of frames on Windows.
// Other lengths fall back to the C library.
// ---------------------------------------------------------------------------

static const double sc_960_v1[64] = {
    0x1p+0, 0.0,
    0x1.fffd3151e5533p-1, 0x1.acedd6862d0d8p-8,
    0x1.fff4c54f76e1cp-1, 0x1.aceb7c72ca0a9p-7,
    0x1.ffe6bc105954ep-1, 0x1.41adacc128e22p-6,
    0x1.ffd315bbf4275p-1, 0x1.ace214390ca91p-6,
    0x1.ffb9d2897136ep-1, 0x1.0c08e3d596aeep-5,
    0x1.ff9af2bfbc297p-1, 0x1.419dcd176e0f7p-5,
    0x1.ff7676b581a63p-1, 0x1.772f2f75f573cp-5,
    0x1.ff4c5ed12e61dp-1, 0x1.acbc748efc90ep-5,
    0x1.ff1cab88edff7p-1, 0x1.e245060be0013p-5,
    0x1.fee75d62a9c46p-1, 0x1.0be426d197a8cp-4,
    0x1.feac74f40720cp-1, 0x1.26a2da8d2974ap-4,
    0x1.fe6bf2e2660afp-1, 0x1.415e532398e49p-4,
    0x1.fe25d7e2df2f8p-1, 0x1.5c164588eb8dbp-4,
    0x1.fdda24ba41f4dp-1, 0x1.76ca66bb0bc83p-4,
    0x1.fd88da3d12526p-1, 0x1.917a6bc29b42dp-4,
    0x1.fd31f94f867c6p-1, 0x1.ac2609b3c576cp-4,
    0x1.fcd582e584632p-1, 0x1.c6ccf5af11fd1p-4,
    0x1.fc7378029f05fp-1, 0x1.e16ee4e236bf9p-4,
    0x1.fc0bd9ba139aep-1, 0x1.fc0b8c88ea05ap-4,
    0x1.fb9ea92ec689bp-1, 0x1.0b5150f6da2d1p-3,
    0x1.fb2be793403b9p-1, 0x1.1899ed3561232p-3,
    0x1.fab39629a9be3p-1, 0x1.25df75b55af15p-3,
    0x1.fa35b643c93b9p-1, 0x1.3321c534bc1bbp-3,
    0x1.f9b24942fe45cp-1, 0x1.4060b67a85375p-3,
    0x1.f92950983df6bp-1, 0x1.4d9c24572b693p-3,
    0x1.f89acdc40ee4bp-1, 0x1.5ad3e9a500cadp-3,
    0x1.f806c25684ea8p-1, 0x1.6807e1489cb34p-3,
    0x1.f76d2fef3cc4bp-1, 0x1.7537e63143e2fp-3,
    0x1.f6ce183d57825p-1, 0x1.8263d35950926p-3,
    0x1.f6297cff75cb0p-1, 0x1.8f8b83c69a60bp-3,
    0x1.f57f6003b2f91p-1, 0x1.9caed28ade228p-3,
};
static const double sc_960_v2[32] = {
    0x1p+0, 0.0,
    0x1.f4cfc327a0080p-1, 0x1.a9cd9ac4258f6p-3,
    0x1.d3bc3aeff7f95p-1, 0x1.a07f921061ad1p-2,
    0x1.9e3779b97f4a8p-1, 0x1.2cf2304755a5ep-1,
    0x1.5698496e20bd8p-1, 0x1.7c7d7a833bec2p-1,
    0x1p-1, 0x1.bb67ae8584caap-1,
    0x1.3c6ef372fe94fp-2, 0x1.e6f0e134454ffp-1,
    0x1.ac2609b3c576cp-4, 0x1.fd31f94f867c6p-1,
    -0x1.ac2609b3c576cp-4, 0x1.fd31f94f867c6p-1,
    -0x1.3c6ef372fe94fp-2, 0x1.e6f0e134454ffp-1,
    -0x1p-1, 0x1.bb67ae8584caap-1,
    -0x1.5698496e20bd8p-1, 0x1.7c7d7a833bec2p-1,
    -0x1.9e3779b97f4a8p-1, 0x1.2cf2304755a5ep-1,
    -0x1.d3bc3aeff7f95p-1, 0x1.a07f921061ad1p-2,
    -0x1.f4cfc327a0080p-1, 0x1.a9cd9ac4258f6p-3,
    -0x1p+0, -0.0,
};
static const double sc_540_v1[64] = {
    0x1p+0, 0.0,
    0x1.fff720a22e0efp-1, 0x1.7d43af0c73423p-7,
    0x1.ffdc82d770fd9p-1, 0x1.7d3d13a8df318p-6,
    0x1.ffb0278bf0567p-1, 0x1.1de58c9f7dc27p-5,
    0x1.ff7210493a3dfp-1, 0x1.7d22a6ca6a97ap-5,
    0x1.ff223f3635ce3p-1, 0x1.dc528b5343a86p-5,
    0x1.fec0b7170fff6p-1, 0x1.1db8f6d6a5128p-4,
    0x1.fe4d7b4d231c4p-1, 0x1.4d3ec0ef1c13fp-4,
    0x1.fdc88fd6d8c45p-1, 0x1.7cb8fe4dc0763p-4,
    0x1.fd31f94f867c6p-1, 0x1.ac2609b3c576cp-4,
    0x1.fc89bcef44cf4p-1, 0x1.db843e577175ep-4,
    0x1.fbcfe08ac0fe9p-1, 0x1.0568fbf959b4fp-3,
    0x1.fb046a930947ap-1, 0x1.1d06c968d9e19p-3,
    0x1.fa27621553bb9p-1, 0x1.349ab5f0b01f0p-3,
    0x1.f938cebabfaefp-1, 0x1.4c23f05ef9113p-3,
    0x1.f838b8c811c17p-1, 0x1.63a1a7e0b7389p-3,
    0x1.f727291d6a810p-1, 0x1.7b130c090fbdbp-3,
    0x1.f6042935f7a9ep-1, 0x1.92774cd883b38p-3,
    0x1.f4cfc327a0080p-1, 0x1.a9cd9ac4258f6p-3,
    0x1.f38a01a2a9fa8p-1, 0x1.c11526bcca962p-3,
    0x1.f232eff15c9e6p-1, 0x1.d84d223638000p-3,
    0x1.f0ca99f79ba25p-1, 0x1.ef74bf2e4b91dp-3,
    0x1.ef510c327dc79p-1, 0x1.0345981a0fb6dp-2,
    0x1.edc653b7de13fp-1, 0x1.0ec7d437946d5p-2,
    0x1.ec2a7e35e7b80p-1, 0x1.1a40add328e29p-2,
    0x1.ea7d99f29cadep-1, 0x1.25afbf23865bep-2,
    0x1.e8bfb5cb57153p-1, 0x1.3114a2b63187bp-2,
    0x1.e6f0e134454ffp-1, 0x1.3c6ef372fe94fp-2,
    0x1.e5112c37e0e51p-1, 0x1.47be4c9f922a2p-2,
    0x1.e320a776602dap-1, 0x1.530249e2df1b0p-2,
    0x1.e11f642522d1cp-1, 0x1.5e3a8748a0bf5p-2,
    0x1.df0d740e19196p-1, 0x1.6966a144d1cbbp-2,
};
static const double sc_540_v2[18] = {
    0x1p+0, 0.0,
    0x1.dceae98f2617bp-1, 0x1.748634b71f8cep-2,
    0x1.787a6c1c246fbp-1, 0x1.5affb0479136ap-1,
    0x1.c0e45dabe05c8p-2, 0x1.cc2ebbb5638cap-1,
    0x1.4d3ec0ef1c13fp-4, 0x1.fe4d7b4d231c4p-1,
    -0x1.25afbf23865bep-2, 0x1.ea7d99f29cadep-1,
    -0x1.3b37fb1bdc939p-1, 0x1.9376253f463d1p-1,
    -0x1.b8657817c9252p-1, 0x1.05244d6538b04p-1,
    -0x1.f938cebabfaefp-1, 0x1.4c23f05ef9113p-3,
};

struct sincos {
    size_t n, mask, shift;
    double *v1, *v2;  // (r, i) pairs
};

static void sc_calc(size_t x, size_t n, double ang, double *out)
{
    x <<= 3;
    if (x < 4 * n) {  // first half
        if (x < 2 * n) {  // first quadrant
            if (x < n) {
                out[0] = cos((double)x * ang);
                out[1] = sin((double)x * ang);
                return;
            }
            out[0] = sin((double)(2 * n - x) * ang);
            out[1] = cos((double)(2 * n - x) * ang);
            return;
        }
        x -= 2 * n;  // second quadrant
        if (x < n) {
            out[0] = -sin((double)x * ang);
            out[1] = cos((double)x * ang);
            return;
        }
        out[0] = -cos((double)(2 * n - x) * ang);
        out[1] = sin((double)(2 * n - x) * ang);
        return;
    }
    x = 8 * n - x;
    if (x < 2 * n) {  // third quadrant
        if (x < n) {
            out[0] = cos((double)x * ang);
            out[1] = -sin((double)x * ang);
            return;
        }
        out[0] = sin((double)(2 * n - x) * ang);
        out[1] = -cos((double)(2 * n - x) * ang);
        return;
    }
    x -= 2 * n;  // fourth quadrant
    if (x < n) {
        out[0] = -sin((double)x * ang);
        out[1] = -cos((double)x * ang);
        return;
    }
    out[0] = -cos((double)(2 * n - x) * ang);
    out[1] = -sin((double)(2 * n - x) * ang);
}

static bool sc_init(struct sincos *s, size_t n)
{
    const long double pi = 3.141592653589793238462643383279502884197L;
    const double ang = (double)(0.25L * pi / (long double)n);
    const size_t nval = (n + 2) / 2;
    s->n = n;
    s->shift = 1;
    while (((size_t)1 << s->shift) * ((size_t)1 << s->shift) < nval)
        s->shift++;
    s->mask = ((size_t)1 << s->shift) - 1;
    const size_t n1 = s->mask + 1, n2 = (nval + s->mask) / (s->mask + 1);
    s->v1 = malloc(2 * n1 * sizeof(double));
    s->v2 = malloc(2 * n2 * sizeof(double));
    if (!s->v1 || !s->v2) {
        free(s->v1);
        free(s->v2);
        return false;
    }
    const double *t1 = NULL, *t2 = NULL;
    if (n == 960 && 2 * n1 == sizeof(sc_960_v1) / sizeof(double) &&
        2 * n2 == sizeof(sc_960_v2) / sizeof(double)) {
        t1 = sc_960_v1;
        t2 = sc_960_v2;
    } else if (n == 540 && 2 * n1 == sizeof(sc_540_v1) / sizeof(double) &&
               2 * n2 == sizeof(sc_540_v2) / sizeof(double)) {
        t1 = sc_540_v1;
        t2 = sc_540_v2;
    }
    if (t1) {
        memcpy(s->v1, t1, 2 * n1 * sizeof(double));
        memcpy(s->v2, t2, 2 * n2 * sizeof(double));
        return true;
    }
    s->v1[0] = 1.0;
    s->v1[1] = 0.0;
    for (size_t i = 1; i < n1; i++)
        sc_calc(i, n, ang, s->v1 + 2 * i);
    s->v2[0] = 1.0;
    s->v2[1] = 0.0;
    for (size_t i = 1; i < n2; i++)
        sc_calc(i * (s->mask + 1), n, ang, s->v2 + 2 * i);
    return true;
}

static void sc_uninit(struct sincos *s)
{
    free(s->v1);
    free(s->v2);
}

// pocketfft cfftp::factorize
static size_t cfactorize(size_t len, size_t *fct)
{
    size_t nf = 0;
    while ((len & 7) == 0) {
        fct[nf++] = 8;
        len >>= 3;
    }
    while ((len & 3) == 0) {
        fct[nf++] = 4;
        len >>= 2;
    }
    if ((len & 1) == 0) {
        len >>= 1;
        // factor 2 should be at the front of the factor list
        fct[nf++] = 2;
        size_t t = fct[0];
        fct[0] = fct[nf - 1];
        fct[nf - 1] = t;
    }
    for (size_t divisor = 3; divisor * divisor <= len; divisor += 2) {
        while ((len % divisor) == 0) {
            fct[nf++] = divisor;
            len /= divisor;
        }
    }
    if (len > 1)
        fct[nf++] = len;
    return nf;
}

// pocketfft rfftp::factorize
static size_t rfactorize(size_t len, size_t *fct)
{
    size_t nf = 0;
    while ((len % 4) == 0) {
        fct[nf++] = 4;
        len >>= 2;
    }
    if ((len % 2) == 0) {
        len >>= 1;
        // factor 2 should be at the front of the factor list
        fct[nf++] = 2;
        size_t t = fct[0];
        fct[0] = fct[nf - 1];
        fct[nf - 1] = t;
    }
    for (size_t divisor = 3; divisor * divisor <= len; divisor += 2) {
        while ((len % divisor) == 0) {
            fct[nf++] = divisor;
            len /= divisor;
        }
    }
    if (len > 1)
        fct[nf++] = len;
    return nf;
}

#define RCF_T float
#define RCF_V rcf_vf
#define RCF_CV rcf_cvf
#define RCF_TW struct rcf_cf
#define RCF_RPLAN struct rcf_rplan_f
#define RCF_CPLAN struct rcf_cplan_f
#define RCF_FN(name) name##_f
#include "vf_animejanai_camera_fft_tmpl.h"
#undef RCF_T
#undef RCF_V
#undef RCF_CV
#undef RCF_TW
#undef RCF_RPLAN
#undef RCF_CPLAN
#undef RCF_FN

#define RCF_T double
#define RCF_V rcf_vd
#define RCF_CV rcf_cvd
#define RCF_TW struct rcf_cd
#define RCF_RPLAN struct rcf_rplan_d
#define RCF_CPLAN struct rcf_cplan_d
#define RCF_FN(name) name##_d
#include "vf_animejanai_camera_fft_tmpl.h"
#undef RCF_T
#undef RCF_V
#undef RCF_CV
#undef RCF_TW
#undef RCF_RPLAN
#undef RCF_CPLAN
#undef RCF_FN
