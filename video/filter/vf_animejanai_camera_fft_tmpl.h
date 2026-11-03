/*
 * Kernel template of vf_animejanai_camera_fft.c, included once per precision
 * (float, double) with RCF_T / RCF_V / RCF_CV / RCF_TW / RCF_FN defined; not
 * a standalone header. Ported from pocketfft_hdronly.h (cfftp pass3/4/5,
 * rfftp radf2-5 / radb2-5, exec, copy_and_norm, factorize, comp_twiddle),
 * https://github.com/mreineck/pocketfft commit
 * 33ae5dc94c9cdc7f1c78346504a85de87cadaa12. Kernel bodies keep pocketfft's
 * names, index macros and operation order: changing the order of any float
 * operation breaks bit parity with numpy.
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

// pocketfft sincos_2pibyn<T>::operator[]: products in double, then to T
static RCF_TW RCF_FN(sc_get)(const struct sincos *s, size_t idx)
{
    const bool upper = 2 * idx > s->n;
    if (upper)
        idx = s->n - idx;
    const double *x1 = s->v1 + 2 * (idx & s->mask);
    const double *x2 = s->v2 + 2 * (idx >> s->shift);
    // volatile: each product rounds to double, as in pocketfft's non-FMA build
    volatile double a = x1[0] * x2[0], b = x1[1] * x2[1];
    volatile double c = x1[0] * x2[1], d = x1[1] * x2[0];
    RCF_TW r = {(RCF_T)(a - b), (RCF_T)(c + d)};
    if (upper)
        r.i = -r.i;
    return r;
}

// ---------------------------------------------------------------------------
// Complex transforms (pocketfft cfftp).
// ---------------------------------------------------------------------------

static inline RCF_CV RCF_FN(cv_add)(RCF_CV a, RCF_CV b)
{
    RCF_CV r;
    r.r = a.r + b.r;
    r.i = a.i + b.i;
    return r;
}

static inline RCF_CV RCF_FN(cv_sub)(RCF_CV a, RCF_CV b)
{
    RCF_CV r;
    r.r = a.r - b.r;
    r.i = a.i - b.i;
    return r;
}

// PM(a, b, c, d): a = c + d; b = c - d
static inline void RCF_FN(cv_pm)(RCF_CV *a, RCF_CV *b, RCF_CV c, RCF_CV d)
{
    *a = RCF_FN(cv_add)(c, d);
    *b = RCF_FN(cv_sub)(c, d);
}

static inline RCF_CV RCF_FN(cv_rotx90)(RCF_CV a, bool fwd)
{
    RCF_CV r;
    if (fwd) {
        r.r = a.i;
        r.i = -a.r;
    } else {
        r.r = -a.i;
        r.i = a.r;
    }
    return r;
}

static inline RCF_CV RCF_FN(cv_special_mul)(RCF_CV v1, RCF_TW v2, bool fwd)
{
    RCF_CV r;
    if (fwd) {
        r.r = v1.r * v2.r + v1.i * v2.i;
        r.i = v1.i * v2.r - v1.r * v2.i;
    } else {
        r.r = v1.r * v2.r - v1.i * v2.i;
        r.i = v1.r * v2.i + v1.i * v2.r;
    }
    return r;
}

#define ADD RCF_FN(cv_add)
#define SUB RCF_FN(cv_sub)
#define CPM RCF_FN(cv_pm)
#define SMUL RCF_FN(cv_special_mul)
#define CH(a, b, c) ch[(a) + ido * ((b) + l1 * (c))]
#define CC(a, b, c) cc[(a) + ido * ((b) + cdim * (c))]
#define WA(x, i) wa[(i) - 1 + (x) * (ido - 1)]

static void RCF_FN(pass3)(size_t ido, size_t l1, const RCF_CV *restrict cc,
                          RCF_CV *restrict ch, const RCF_TW *restrict wa,
                          bool fwd)
{
    const size_t cdim = 3;
    const RCF_T tw1r = (RCF_T)-0.5,
                tw1i = (fwd ? (RCF_T)-1 : (RCF_T)1) *
                       (RCF_T)0.8660254037844386467637231707529362L;

    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 0; i < ido; i++) {
            // POCKETFFT_PREP3(i)
            RCF_CV t0 = CC(i, 0, k), t1, t2;
            CPM(&t1, &t2, CC(i, 1, k), CC(i, 2, k));
            CH(i, k, 0) = ADD(t0, t1);
            RCF_CV ca, cb;
            ca.r = t0.r + t1.r * tw1r;
            ca.i = t0.i + t1.i * tw1r;
            cb.r = -t2.i * tw1i;
            cb.i = t2.r * tw1i;
            if (i == 0) {  // POCKETFFT_PARTSTEP3a(1, 2, tw1r, tw1i)
                CPM(&CH(0, k, 1), &CH(0, k, 2), ca, cb);
            } else {  // POCKETFFT_PARTSTEP3b(1, 2, tw1r, tw1i)
                CH(i, k, 1) = SMUL(ADD(ca, cb), WA(0, i), fwd);
                CH(i, k, 2) = SMUL(SUB(ca, cb), WA(1, i), fwd);
            }
        }
    }
}

static void RCF_FN(pass4)(size_t ido, size_t l1, const RCF_CV *restrict cc,
                          RCF_CV *restrict ch, const RCF_TW *restrict wa,
                          bool fwd)
{
    const size_t cdim = 4;

    for (size_t k = 0; k < l1; k++) {
        {
            RCF_CV t1, t2, t3, t4;
            CPM(&t2, &t1, CC(0, 0, k), CC(0, 2, k));
            CPM(&t3, &t4, CC(0, 1, k), CC(0, 3, k));
            t4 = RCF_FN(cv_rotx90)(t4, fwd);
            CPM(&CH(0, k, 0), &CH(0, k, 2), t2, t3);
            CPM(&CH(0, k, 1), &CH(0, k, 3), t1, t4);
        }
        for (size_t i = 1; i < ido; i++) {
            RCF_CV t1, t2, t3, t4;
            RCF_CV cc0 = CC(i, 0, k), cc1 = CC(i, 1, k), cc2 = CC(i, 2, k),
                   cc3 = CC(i, 3, k);
            CPM(&t2, &t1, cc0, cc2);
            CPM(&t3, &t4, cc1, cc3);
            t4 = RCF_FN(cv_rotx90)(t4, fwd);
            CH(i, k, 0) = ADD(t2, t3);
            CH(i, k, 1) = SMUL(ADD(t1, t4), WA(0, i), fwd);
            CH(i, k, 2) = SMUL(SUB(t2, t3), WA(1, i), fwd);
            CH(i, k, 3) = SMUL(SUB(t1, t4), WA(2, i), fwd);
        }
    }
}

static void RCF_FN(pass5)(size_t ido, size_t l1, const RCF_CV *restrict cc,
                          RCF_CV *restrict ch, const RCF_TW *restrict wa,
                          bool fwd)
{
    const size_t cdim = 5;
    const RCF_T sg = fwd ? (RCF_T)-1 : (RCF_T)1;
    const RCF_T tw1r = (RCF_T)0.3090169943749474241022934171828191L,
                tw1i = sg * (RCF_T)0.9510565162951535721164393333793821L,
                tw2r = (RCF_T)-0.8090169943749474241022934171828191L,
                tw2i = sg * (RCF_T)0.5877852522924731291687059546390728L;

    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 0; i < ido; i++) {
            // POCKETFFT_PREP5(i)
            RCF_CV t0 = CC(i, 0, k), t1, t2, t3, t4;
            CPM(&t1, &t4, CC(i, 1, k), CC(i, 4, k));
            CPM(&t2, &t3, CC(i, 2, k), CC(i, 3, k));
            CH(i, k, 0).r = t0.r + t1.r + t2.r;
            CH(i, k, 0).i = t0.i + t1.i + t2.i;
            RCF_CV ca, cb;
            // PARTSTEP5(1, 4, tw1r, tw2r, +tw1i, +tw2i)
            ca.r = t0.r + tw1r * t1.r + tw2r * t2.r;
            ca.i = t0.i + tw1r * t1.i + tw2r * t2.i;
            cb.i = tw1i * t4.r + tw2i * t3.r;
            cb.r = -(tw1i * t4.i + tw2i * t3.i);
            if (i == 0) {
                CPM(&CH(0, k, 1), &CH(0, k, 4), ca, cb);
            } else {
                CH(i, k, 1) = SMUL(ADD(ca, cb), WA(0, i), fwd);
                CH(i, k, 4) = SMUL(SUB(ca, cb), WA(3, i), fwd);
            }
            // PARTSTEP5(2, 3, tw2r, tw1r, +tw2i, -tw1i)
            ca.r = t0.r + tw2r * t1.r + tw1r * t2.r;
            ca.i = t0.i + tw2r * t1.i + tw1r * t2.i;
            cb.i = tw2i * t4.r - tw1i * t3.r;
            cb.r = -(tw2i * t4.i - tw1i * t3.i);
            if (i == 0) {
                CPM(&CH(0, k, 2), &CH(0, k, 3), ca, cb);
            } else {
                CH(i, k, 2) = SMUL(ADD(ca, cb), WA(1, i), fwd);
                CH(i, k, 3) = SMUL(SUB(ca, cb), WA(2, i), fwd);
            }
        }
    }
}

#undef ADD
#undef SUB
#undef CPM
#undef SMUL
#undef CH
#undef CC
#undef WA

bool RCF_FN(rcf_cplan_init)(RCF_CPLAN *p, size_t n)
{
    memset(p, 0, sizeof(*p));
    // pocketfft_c picks cfftp when the largest prime factor squared is <= n
    // (true for the factors below)
    if (n < 2 || n > ((size_t)1 << 24))
        return false;
    p->n = n;
    p->nfct = cfactorize(n, p->fct);
    size_t twsize = 0, l1 = 1;
    for (size_t k = 0; k < p->nfct; k++) {
        size_t ip = p->fct[k];
        if (ip != 3 && ip != 4 && ip != 5)  // pass2/7/8/11/g not ported
            return false;
        twsize += (ip - 1) * (n / (l1 * ip) - 1);
        l1 *= ip;
    }
    struct sincos sc;
    p->mem = malloc((twsize ? twsize : 1) * sizeof(RCF_TW));
    if (!p->mem || !sc_init(&sc, n)) {
        free(p->mem);
        p->mem = NULL;
        return false;
    }
    l1 = 1;
    size_t memofs = 0;
    for (size_t k = 0; k < p->nfct; k++) {
        size_t ip = p->fct[k], ido = n / (l1 * ip);
        RCF_TW *tw = p->mem + memofs;
        p->tw[k] = tw;
        memofs += (ip - 1) * (ido - 1);
        for (size_t j = 1; j < ip; j++)
            for (size_t i = 1; i < ido; i++)
                tw[(j - 1) * (ido - 1) + i - 1] = RCF_FN(sc_get)(&sc, j * l1 * i);
        l1 *= ip;
    }
    sc_uninit(&sc);
    return true;
}

void RCF_FN(rcf_cplan_uninit)(RCF_CPLAN *p)
{
    free(p->mem);
    memset(p, 0, sizeof(*p));
}

void RCF_FN(rcf_cfft)(const RCF_CPLAN *p, RCF_CV *c, RCF_CV *ch, bool forward,
                      RCF_T fct)
{
    size_t l1 = 1;
    RCF_CV *p1 = c, *p2 = ch;
    for (size_t k1 = 0; k1 < p->nfct; k1++) {
        size_t ip = p->fct[k1];
        size_t l2 = ip * l1;
        size_t ido = p->n / l2;
        if (ip == 4)
            RCF_FN(pass4)(ido, l1, p1, p2, p->tw[k1], forward);
        else if (ip == 3)
            RCF_FN(pass3)(ido, l1, p1, p2, p->tw[k1], forward);
        else
            RCF_FN(pass5)(ido, l1, p1, p2, p->tw[k1], forward);
        RCF_CV *t = p1;
        p1 = p2;
        p2 = t;
        l1 = l2;
    }
    if (p1 != c) {
        if (fct != 1) {
            for (size_t i = 0; i < p->n; i++) {
                c[i].r = ch[i].r * fct;
                c[i].i = ch[i].i * fct;
            }
        } else {
            memcpy(c, p1, p->n * sizeof(*c));
        }
    } else if (fct != 1) {
        for (size_t i = 0; i < p->n; i++) {
            c[i].r *= fct;
            c[i].i *= fct;
        }
    }
}

// ---------------------------------------------------------------------------
// Real transforms (pocketfft rfftp).
// ---------------------------------------------------------------------------

// PM(a, b, c, d): a = c + d; b = c - d
#define PM(a, b, c, d)          \
    do {                        \
        RCF_V pm_c_ = (c);      \
        RCF_V pm_d_ = (d);      \
        (a) = pm_c_ + pm_d_;    \
        (b) = pm_c_ - pm_d_;    \
    } while (0)

// (a+ib) = conj(c+id) * (e+if): a = c*e + d*f; b = c*f - d*e
// (no call site has a or b among c..f, so no temporaries are needed)
#define MULPM(a, b, c, d, e, f)              \
    do {                                     \
        (a) = (c) * (e) + (d) * (f);         \
        (b) = (c) * (f) - (d) * (e);         \
    } while (0)

// a2=a+b; b2=i*(b-a);
#define REARRANGE(rx, ix, ry, iy)                                    \
    do {                                                             \
        RCF_V t1_ = rx + ry, t2_ = ry - rx, t3_ = ix + iy,           \
              t4_ = ix - iy;                                         \
        rx = t1_;                                                    \
        ix = t3_;                                                    \
        ry = t4_;                                                    \
        iy = t2_;                                                    \
    } while (0)

#define WA(x, i) wa[(i) + (x) * (ido - 1)]

// forward kernels: CC is [ido][l1][ip], CH is [ido][ip][l1]
#define CC(a, b, c) cc[(a) + ido * ((b) + l1 * (c))]
#define CH(a, b, c) ch[(a) + ido * ((b) + cdim * (c))]

static void RCF_FN(radf2)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 2;

    for (size_t k = 0; k < l1; k++)
        PM(CH(0, 0, k), CH(ido - 1, 1, k), CC(0, k, 0), CC(0, k, 1));
    if ((ido & 1) == 0) {
        for (size_t k = 0; k < l1; k++) {
            CH(0, 1, k) = -CC(ido - 1, k, 1);
            CH(ido - 1, 0, k) = CC(ido - 1, k, 0);
        }
    }
    if (ido <= 2)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2; i < ido; i += 2) {
            size_t ic = ido - i;
            RCF_V tr2, ti2;
            MULPM(tr2, ti2, WA(0, i - 2), WA(0, i - 1), CC(i - 1, k, 1),
                  CC(i, k, 1));
            PM(CH(i - 1, 0, k), CH(ic - 1, 1, k), CC(i - 1, k, 0), tr2);
            PM(CH(i, 0, k), CH(ic, 1, k), ti2, CC(i, k, 0));
        }
    }
}

static void RCF_FN(radf3)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 3;
    const RCF_T taur = (RCF_T)-0.5,
                taui = (RCF_T)0.8660254037844386467637231707529362L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V cr2 = CC(0, k, 1) + CC(0, k, 2);
        CH(0, 0, k) = CC(0, k, 0) + cr2;
        CH(0, 2, k) = taui * (CC(0, k, 2) - CC(0, k, 1));
        CH(ido - 1, 1, k) = CC(0, k, 0) + taur * cr2;
    }
    if (ido == 1)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2; i < ido; i += 2) {
            size_t ic = ido - i;
            RCF_V di2, di3, dr2, dr3;
            MULPM(dr2, di2, WA(0, i - 2), WA(0, i - 1), CC(i - 1, k, 1),
                  CC(i, k, 1));
            MULPM(dr3, di3, WA(1, i - 2), WA(1, i - 1), CC(i - 1, k, 2),
                  CC(i, k, 2));
            REARRANGE(dr2, di2, dr3, di3);
            CH(i - 1, 0, k) = CC(i - 1, k, 0) + dr2;
            CH(i, 0, k) = CC(i, k, 0) + di2;
            RCF_V tr2 = CC(i - 1, k, 0) + taur * dr2;
            RCF_V ti2 = CC(i, k, 0) + taur * di2;
            RCF_V tr3 = taui * dr3;
            RCF_V ti3 = taui * di3;
            PM(CH(i - 1, 2, k), CH(ic - 1, 1, k), tr2, tr3);
            PM(CH(i, 2, k), CH(ic, 1, k), ti3, ti2);
        }
    }
}

static void RCF_FN(radf4)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 4;
    const RCF_T hsqt2 = (RCF_T)0.707106781186547524400844362104849L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V tr1, tr2;
        PM(tr1, CH(0, 2, k), CC(0, k, 3), CC(0, k, 1));
        PM(tr2, CH(ido - 1, 1, k), CC(0, k, 0), CC(0, k, 2));
        PM(CH(0, 0, k), CH(ido - 1, 3, k), tr2, tr1);
    }
    if ((ido & 1) == 0) {
        for (size_t k = 0; k < l1; k++) {
            RCF_V ti1 = -hsqt2 * (CC(ido - 1, k, 1) + CC(ido - 1, k, 3));
            RCF_V tr1 = hsqt2 * (CC(ido - 1, k, 1) - CC(ido - 1, k, 3));
            PM(CH(ido - 1, 0, k), CH(ido - 1, 2, k), CC(ido - 1, k, 0), tr1);
            PM(CH(0, 3, k), CH(0, 1, k), ti1, CC(ido - 1, k, 2));
        }
    }
    if (ido <= 2)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2; i < ido; i += 2) {
            size_t ic = ido - i;
            RCF_V ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2,
                  tr3, tr4;
            MULPM(cr2, ci2, WA(0, i - 2), WA(0, i - 1), CC(i - 1, k, 1),
                  CC(i, k, 1));
            MULPM(cr3, ci3, WA(1, i - 2), WA(1, i - 1), CC(i - 1, k, 2),
                  CC(i, k, 2));
            MULPM(cr4, ci4, WA(2, i - 2), WA(2, i - 1), CC(i - 1, k, 3),
                  CC(i, k, 3));
            PM(tr1, tr4, cr4, cr2);
            PM(ti1, ti4, ci2, ci4);
            PM(tr2, tr3, CC(i - 1, k, 0), cr3);
            PM(ti2, ti3, CC(i, k, 0), ci3);
            PM(CH(i - 1, 0, k), CH(ic - 1, 3, k), tr2, tr1);
            PM(CH(i, 0, k), CH(ic, 3, k), ti1, ti2);
            PM(CH(i - 1, 2, k), CH(ic - 1, 1, k), tr3, ti4);
            PM(CH(i, 2, k), CH(ic, 1, k), tr4, ti3);
        }
    }
}

static void RCF_FN(radf5)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 5;
    const RCF_T tr11 = (RCF_T)0.3090169943749474241022934171828191L,
                ti11 = (RCF_T)0.9510565162951535721164393333793821L,
                tr12 = (RCF_T)-0.8090169943749474241022934171828191L,
                ti12 = (RCF_T)0.5877852522924731291687059546390728L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V cr2, cr3, ci4, ci5;
        PM(cr2, ci5, CC(0, k, 4), CC(0, k, 1));
        PM(cr3, ci4, CC(0, k, 3), CC(0, k, 2));
        CH(0, 0, k) = CC(0, k, 0) + cr2 + cr3;
        CH(ido - 1, 1, k) = CC(0, k, 0) + tr11 * cr2 + tr12 * cr3;
        CH(0, 2, k) = ti11 * ci5 + ti12 * ci4;
        CH(ido - 1, 3, k) = CC(0, k, 0) + tr12 * cr2 + tr11 * cr3;
        CH(0, 4, k) = ti12 * ci5 - ti11 * ci4;
    }
    if (ido == 1)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
            RCF_V di2, di3, di4, di5, dr2, dr3, dr4, dr5;
            MULPM(dr2, di2, WA(0, i - 2), WA(0, i - 1), CC(i - 1, k, 1),
                  CC(i, k, 1));
            MULPM(dr3, di3, WA(1, i - 2), WA(1, i - 1), CC(i - 1, k, 2),
                  CC(i, k, 2));
            MULPM(dr4, di4, WA(2, i - 2), WA(2, i - 1), CC(i - 1, k, 3),
                  CC(i, k, 3));
            MULPM(dr5, di5, WA(3, i - 2), WA(3, i - 1), CC(i - 1, k, 4),
                  CC(i, k, 4));
            REARRANGE(dr2, di2, dr5, di5);
            REARRANGE(dr3, di3, dr4, di4);
            CH(i - 1, 0, k) = CC(i - 1, k, 0) + dr2 + dr3;
            CH(i, 0, k) = CC(i, k, 0) + di2 + di3;
            RCF_V tr2 = CC(i - 1, k, 0) + tr11 * dr2 + tr12 * dr3;
            RCF_V ti2 = CC(i, k, 0) + tr11 * di2 + tr12 * di3;
            RCF_V tr3 = CC(i - 1, k, 0) + tr12 * dr2 + tr11 * dr3;
            RCF_V ti3 = CC(i, k, 0) + tr12 * di2 + tr11 * di3;
            RCF_V tr5 = ti11 * dr5 + ti12 * dr4;
            RCF_V ti5 = ti11 * di5 + ti12 * di4;
            RCF_V tr4 = ti12 * dr5 - ti11 * dr4;
            RCF_V ti4 = ti12 * di5 - ti11 * di4;
            PM(CH(i - 1, 2, k), CH(ic - 1, 1, k), tr2, tr5);
            PM(CH(i, 2, k), CH(ic, 1, k), ti5, ti2);
            PM(CH(i - 1, 4, k), CH(ic - 1, 3, k), tr3, tr4);
            PM(CH(i, 4, k), CH(ic, 3, k), ti4, ti3);
        }
    }
}

#undef CC
#undef CH

// backward kernels: CC is [ido][ip][l1], CH is [ido][l1][ip]
#define CC(a, b, c) cc[(a) + ido * ((b) + cdim * (c))]
#define CH(a, b, c) ch[(a) + ido * ((b) + l1 * (c))]

static void RCF_FN(radb2)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 2;

    for (size_t k = 0; k < l1; k++)
        PM(CH(0, k, 0), CH(0, k, 1), CC(0, 0, k), CC(ido - 1, 1, k));
    if ((ido & 1) == 0) {
        for (size_t k = 0; k < l1; k++) {
            CH(ido - 1, k, 0) = (RCF_T)2 * CC(ido - 1, 0, k);
            CH(ido - 1, k, 1) = (RCF_T)-2 * CC(0, 1, k);
        }
    }
    if (ido <= 2)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2; i < ido; i += 2) {
            size_t ic = ido - i;
            RCF_V ti2, tr2;
            PM(CH(i - 1, k, 0), tr2, CC(i - 1, 0, k), CC(ic - 1, 1, k));
            PM(ti2, CH(i, k, 0), CC(i, 0, k), CC(ic, 1, k));
            MULPM(CH(i, k, 1), CH(i - 1, k, 1), WA(0, i - 2), WA(0, i - 1),
                  ti2, tr2);
        }
    }
}

static void RCF_FN(radb3)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 3;
    const RCF_T taur = (RCF_T)-0.5,
                taui = (RCF_T)0.8660254037844386467637231707529362L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V tr2 = (RCF_T)2 * CC(ido - 1, 1, k);
        RCF_V cr2 = CC(0, 0, k) + taur * tr2;
        CH(0, k, 0) = CC(0, 0, k) + tr2;
        RCF_V ci3 = ((RCF_T)2 * taui) * CC(0, 2, k);
        PM(CH(0, k, 2), CH(0, k, 1), cr2, ci3);
    }
    if (ido == 1)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
            RCF_V tr2 = CC(i - 1, 2, k) + CC(ic - 1, 1, k);
            RCF_V ti2 = CC(i, 2, k) - CC(ic, 1, k);
            RCF_V cr2 = CC(i - 1, 0, k) + taur * tr2;
            RCF_V ci2 = CC(i, 0, k) + taur * ti2;
            CH(i - 1, k, 0) = CC(i - 1, 0, k) + tr2;
            CH(i, k, 0) = CC(i, 0, k) + ti2;
            RCF_V cr3 = taui * (CC(i - 1, 2, k) - CC(ic - 1, 1, k));
            RCF_V ci3 = taui * (CC(i, 2, k) + CC(ic, 1, k));
            RCF_V di2, di3, dr2, dr3;
            PM(dr3, dr2, cr2, ci3);
            PM(di2, di3, ci2, cr3);
            MULPM(CH(i, k, 1), CH(i - 1, k, 1), WA(0, i - 2), WA(0, i - 1),
                  di2, dr2);
            MULPM(CH(i, k, 2), CH(i - 1, k, 2), WA(1, i - 2), WA(1, i - 1),
                  di3, dr3);
        }
    }
}

static void RCF_FN(radb4)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 4;
    const RCF_T sqrt2 = (RCF_T)1.414213562373095048801688724209698L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V tr1, tr2;
        PM(tr2, tr1, CC(0, 0, k), CC(ido - 1, 3, k));
        RCF_V tr3 = (RCF_T)2 * CC(ido - 1, 1, k);
        RCF_V tr4 = (RCF_T)2 * CC(0, 2, k);
        PM(CH(0, k, 0), CH(0, k, 2), tr2, tr3);
        PM(CH(0, k, 3), CH(0, k, 1), tr1, tr4);
    }
    if ((ido & 1) == 0) {
        for (size_t k = 0; k < l1; k++) {
            RCF_V tr1, tr2, ti1, ti2;
            PM(ti1, ti2, CC(0, 3, k), CC(0, 1, k));
            PM(tr2, tr1, CC(ido - 1, 0, k), CC(ido - 1, 2, k));
            CH(ido - 1, k, 0) = tr2 + tr2;
            CH(ido - 1, k, 1) = sqrt2 * (tr1 - ti1);
            CH(ido - 1, k, 2) = ti2 + ti2;
            CH(ido - 1, k, 3) = -sqrt2 * (tr1 + ti1);
        }
    }
    if (ido <= 2)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2; i < ido; i += 2) {
            RCF_V ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2,
                  tr3, tr4;
            size_t ic = ido - i;
            PM(tr2, tr1, CC(i - 1, 0, k), CC(ic - 1, 3, k));
            PM(ti1, ti2, CC(i, 0, k), CC(ic, 3, k));
            PM(tr4, ti3, CC(i, 2, k), CC(ic, 1, k));
            PM(tr3, ti4, CC(i - 1, 2, k), CC(ic - 1, 1, k));
            PM(CH(i - 1, k, 0), cr3, tr2, tr3);
            PM(CH(i, k, 0), ci3, ti2, ti3);
            PM(cr4, cr2, tr1, tr4);
            PM(ci2, ci4, ti1, ti4);
            MULPM(CH(i, k, 1), CH(i - 1, k, 1), WA(0, i - 2), WA(0, i - 1),
                  ci2, cr2);
            MULPM(CH(i, k, 2), CH(i - 1, k, 2), WA(1, i - 2), WA(1, i - 1),
                  ci3, cr3);
            MULPM(CH(i, k, 3), CH(i - 1, k, 3), WA(2, i - 2), WA(2, i - 1),
                  ci4, cr4);
        }
    }
}

static void RCF_FN(radb5)(size_t ido, size_t l1, const RCF_V *restrict cc,
                          RCF_V *restrict ch, const RCF_T *restrict wa)
{
    const size_t cdim = 5;
    const RCF_T tr11 = (RCF_T)0.3090169943749474241022934171828191L,
                ti11 = (RCF_T)0.9510565162951535721164393333793821L,
                tr12 = (RCF_T)-0.8090169943749474241022934171828191L,
                ti12 = (RCF_T)0.5877852522924731291687059546390728L;

    for (size_t k = 0; k < l1; k++) {
        RCF_V ti5 = CC(0, 2, k) + CC(0, 2, k);
        RCF_V ti4 = CC(0, 4, k) + CC(0, 4, k);
        RCF_V tr2 = CC(ido - 1, 1, k) + CC(ido - 1, 1, k);
        RCF_V tr3 = CC(ido - 1, 3, k) + CC(ido - 1, 3, k);
        CH(0, k, 0) = CC(0, 0, k) + tr2 + tr3;
        RCF_V cr2 = CC(0, 0, k) + tr11 * tr2 + tr12 * tr3;
        RCF_V cr3 = CC(0, 0, k) + tr12 * tr2 + tr11 * tr3;
        RCF_V ci4, ci5;
        MULPM(ci5, ci4, ti5, ti4, ti11, ti12);
        PM(CH(0, k, 4), CH(0, k, 1), cr2, ci5);
        PM(CH(0, k, 3), CH(0, k, 2), cr3, ci4);
    }
    if (ido == 1)
        return;
    for (size_t k = 0; k < l1; k++) {
        for (size_t i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
            RCF_V tr2, tr3, tr4, tr5, ti2, ti3, ti4, ti5;
            PM(tr2, tr5, CC(i - 1, 2, k), CC(ic - 1, 1, k));
            PM(ti5, ti2, CC(i, 2, k), CC(ic, 1, k));
            PM(tr3, tr4, CC(i - 1, 4, k), CC(ic - 1, 3, k));
            PM(ti4, ti3, CC(i, 4, k), CC(ic, 3, k));
            CH(i - 1, k, 0) = CC(i - 1, 0, k) + tr2 + tr3;
            CH(i, k, 0) = CC(i, 0, k) + ti2 + ti3;
            RCF_V cr2 = CC(i - 1, 0, k) + tr11 * tr2 + tr12 * tr3;
            RCF_V ci2 = CC(i, 0, k) + tr11 * ti2 + tr12 * ti3;
            RCF_V cr3 = CC(i - 1, 0, k) + tr12 * tr2 + tr11 * tr3;
            RCF_V ci3 = CC(i, 0, k) + tr12 * ti2 + tr11 * ti3;
            RCF_V ci4, ci5, cr5, cr4;
            MULPM(cr5, cr4, tr5, tr4, ti11, ti12);
            MULPM(ci5, ci4, ti5, ti4, ti11, ti12);
            RCF_V dr2, dr3, dr4, dr5, di2, di3, di4, di5;
            PM(dr4, dr3, cr3, ci4);
            PM(di3, di4, ci3, cr4);
            PM(dr5, dr2, cr2, ci5);
            PM(di2, di5, ci2, cr5);
            MULPM(CH(i, k, 1), CH(i - 1, k, 1), WA(0, i - 2), WA(0, i - 1),
                  di2, dr2);
            MULPM(CH(i, k, 2), CH(i - 1, k, 2), WA(1, i - 2), WA(1, i - 1),
                  di3, dr3);
            MULPM(CH(i, k, 3), CH(i - 1, k, 3), WA(2, i - 2), WA(2, i - 1),
                  di4, dr4);
            MULPM(CH(i, k, 4), CH(i - 1, k, 4), WA(3, i - 2), WA(3, i - 1),
                  di5, dr5);
        }
    }
}

#undef CC
#undef CH
#undef WA
#undef PM
#undef MULPM
#undef REARRANGE

bool RCF_FN(rcf_rplan_init)(RCF_RPLAN *p, size_t n)
{
    memset(p, 0, sizeof(*p));
    if (n < 2 || n > ((size_t)1 << 24))
        return false;
    p->n = n;
    p->nfct = rfactorize(n, p->fct);
    size_t twsize = 0, l1 = 1;
    for (size_t k = 0; k < p->nfct; k++) {
        size_t ip = p->fct[k];
        if (ip < 2 || ip > 5)  // radfg / radbg not ported
            return false;
        twsize += (ip - 1) * (n / (l1 * ip) - 1);
        l1 *= ip;
    }
    struct sincos sc;
    p->mem = malloc((twsize ? twsize : 1) * sizeof(RCF_T));
    if (!p->mem || !sc_init(&sc, n)) {
        free(p->mem);
        p->mem = NULL;
        return false;
    }
    l1 = 1;
    RCF_T *ptr = p->mem;
    for (size_t k = 0; k < p->nfct; k++) {
        size_t ip = p->fct[k], ido = n / (l1 * ip);
        if (k < p->nfct - 1) {  // last factor doesn't need twiddles
            RCF_T *tw = ptr;
            p->tw[k] = tw;
            ptr += (ip - 1) * (ido - 1);
            for (size_t j = 1; j < ip; j++) {
                for (size_t i = 1; i <= (ido - 1) / 2; i++) {
                    RCF_TW t = RCF_FN(sc_get)(&sc, j * l1 * i);
                    tw[(j - 1) * (ido - 1) + 2 * i - 2] = t.r;
                    tw[(j - 1) * (ido - 1) + 2 * i - 1] = t.i;
                }
            }
        }
        l1 *= ip;
    }
    sc_uninit(&sc);
    return true;
}

void RCF_FN(rcf_rplan_uninit)(RCF_RPLAN *p)
{
    free(p->mem);
    memset(p, 0, sizeof(*p));
}

void RCF_FN(rcf_rfft)(const RCF_RPLAN *p, RCF_V *c, RCF_V *ch, bool forward,
                      RCF_T fct)
{
    const size_t n = p->n, nf = p->nfct;
    RCF_V *p1 = c, *p2 = ch;
    if (forward) {
        for (size_t k1 = 0, l1 = n; k1 < nf; k1++) {
            size_t k = nf - k1 - 1;
            size_t ip = p->fct[k];
            size_t ido = n / l1;
            l1 /= ip;
            if (ip == 4)
                RCF_FN(radf4)(ido, l1, p1, p2, p->tw[k]);
            else if (ip == 2)
                RCF_FN(radf2)(ido, l1, p1, p2, p->tw[k]);
            else if (ip == 3)
                RCF_FN(radf3)(ido, l1, p1, p2, p->tw[k]);
            else
                RCF_FN(radf5)(ido, l1, p1, p2, p->tw[k]);
            RCF_V *t = p1;
            p1 = p2;
            p2 = t;
        }
    } else {
        for (size_t k = 0, l1 = 1; k < nf; k++) {
            size_t ip = p->fct[k], ido = n / (ip * l1);
            if (ip == 4)
                RCF_FN(radb4)(ido, l1, p1, p2, p->tw[k]);
            else if (ip == 2)
                RCF_FN(radb2)(ido, l1, p1, p2, p->tw[k]);
            else if (ip == 3)
                RCF_FN(radb3)(ido, l1, p1, p2, p->tw[k]);
            else
                RCF_FN(radb5)(ido, l1, p1, p2, p->tw[k]);
            RCF_V *t = p1;
            p1 = p2;
            p2 = t;
            l1 *= ip;
        }
    }
    // copy_and_norm
    if (p1 != c) {
        if (fct != 1) {
            for (size_t i = 0; i < n; i++)
                c[i] = fct * p1[i];
        } else {
            memcpy(c, p1, n * sizeof(*c));
        }
    } else if (fct != 1) {
        for (size_t i = 0; i < n; i++)
            c[i] *= fct;
    }
}
