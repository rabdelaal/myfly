/**
 * ============================================================================
 * htc_v21.c — Audit, vérification et benchmark de HTC-Core v2.1
 * (offset +128 bijectif, accumulateur signé unique, GEMV AVX2 4×64)
 *
 *   1. Référence scalaire int32
 *   2. HTC v2.0 scalaire (offset 127 + clamping, extraction de bits) — baseline
 *   3. HTC v2.1 GEMV pré-décompacté (masques AVX2 pré-générés)
 *   4. HTC v2.1 GEMV bitplane online (2 bits/poids, décompactage à chaud)
 *   5. Vérification : 500 000 vecteurs a ∈ [-128, 127] — le défaut -128 de
 *      v2.0 doit avoir DISPARU avec l'offset bijectif +128
 *   6. Microbenchmarks ns / 64 MACs
 *
 * Compilation : gcc -O3 -march=native -o htc_v21 htc_v21.c
 * ============================================================================
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define DIM 64

/* ---------- Chronomètre ---------- */
static double now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
}

/* ---------- RNG ---------- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static uint32_t rng_below(uint32_t m) { return (uint32_t)(rng_next() % m); }

/* ---------- 3. Décompactage AVX2 32 bits -> 32 octets (0xFF / 0x00) ---------- */
static inline __m256i htc_expand_mask32_pure_avx2(uint32_t mask32) {
    __m256i v = _mm256_set1_epi32((int)mask32);
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1,
        2,2,2,2, 2,2,2,2, 3,3,3,3, 3,3,3,3);
    __m256i bytes = _mm256_shuffle_epi8(v, shuf);
    const __m256i bit_mask = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128);
    __m256i tested = _mm256_and_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(tested, bit_mask);
}

/* ---------- Réduction horizontale 8 × int64 ---------- */
static inline int64_t hsum_epi64_avx2(__m256i v) {
    __m128i low  = _mm256_castsi256_si128(v);
    __m128i high = _mm256_extracti128_si256(v, 1);
    __m128i sum  = _mm_add_epi64(low, high);
    __m128i hi64 = _mm_unpackhi_epi64(sum, sum);
    return _mm_cvtsi128_si64(_mm_add_epi64(sum, hi64));
}

/* ---------- 1. Référence scalaire (a ∈ [-128,127], int32) ---------- */
static int32_t ref_dot(const int8_t *w, const int8_t *a) {
    int32_t s = 0;
    for (int k = 0; k < DIM; ++k) s += (int32_t)w[k] * (int32_t)a[k];
    return s;
}

/* ---------- 2. HTC v2.0 scalaire (offset 127 + clamp) ---------- */
typedef struct { uint64_t mask_pos, mask_neg; int32_t pop_bias; } htc20_row_t;

static void htc20_pack(const int8_t *w, htc20_row_t *r) {
    uint64_t mp = 0, mn = 0; int32_t pp = 0, pn = 0;
    for (int k = 0; k < DIM; ++k) {
        uint64_t bp = (w[k] == 1), bn = (w[k] == -1);
        mp |= bp << k; mn |= bn << k; pp += (int32_t)bp; pn += (int32_t)bn;
    }
    r->mask_pos = mp; r->mask_neg = mn; r->pop_bias = 127 * (pp - pn);
}

static int32_t htc20_compute(const htc20_row_t *r, const uint8_t *x127) {
    int32_t sp = 0, sn = 0;
    for (int k = 0; k < DIM; ++k) {
        sp += (int32_t)(x127[k] * (uint32_t)((r->mask_pos >> k) & 1ULL));
        sn += (int32_t)(x127[k] * (uint32_t)((r->mask_neg >> k) & 1ULL));
    }
    return (sp - sn) - r->pop_bias;
}

/* ---------- v2.1 : bitplanes + terme statique + masques pré-décompactés ---- */
typedef struct {
    __m256i wp[2], wn[2];       /* masques pré-décompactés */
    uint32_t pp[2], pn[2];      /* bitplanes compacts (2 bits/poids) */
    int32_t c_row;              /* 128*(Npos - Nneg) - Bias */
} htc21_row_t;

static void htc21_pack(const int8_t *w, htc21_row_t *r) {
    uint32_t pp[2] = {0, 0}, pn[2] = {0, 0};
    for (int k = 0; k < DIM; ++k) {
        if (w[k] == 1)  pp[k >> 5] |= 1u << (k & 31);
        if (w[k] == -1) pn[k >> 5] |= 1u << (k & 31);
    }
    int npos = __builtin_popcount(pp[0]) + __builtin_popcount(pp[1]);
    int nneg = __builtin_popcount(pn[0]) + __builtin_popcount(pn[1]);
    r->pp[0] = pp[0]; r->pp[1] = pp[1]; r->pn[0] = pn[0]; r->pn[1] = pn[1];
    r->c_row = 128 * (npos - nneg);   /* Bias = 0 dans ce harness */
    r->wp[0] = htc_expand_mask32_pure_avx2(pp[0]);
    r->wp[1] = htc_expand_mask32_pure_avx2(pp[1]);
    r->wn[0] = htc_expand_mask32_pure_avx2(pn[0]);
    r->wn[1] = htc_expand_mask32_pure_avx2(pn[1]);
}

/* ---------- 3. GEMV v2.1 4 lignes × 64, masques pré-décompactés ---------- */
static void htc_gemv_v21_4x64(const uint8_t *u, const __m256i *w_pos,
                              const __m256i *w_neg, const int32_t *c_row,
                              int32_t *y) {
    __m256i u0 = _mm256_loadu_si256((const __m256i *)(u));
    __m256i u1 = _mm256_loadu_si256((const __m256i *)(u + 32));
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc[4];
    for (int r = 0; r < 4; r++) {
        __m256i p0 = _mm256_sad_epu8(_mm256_and_si256(u0, w_pos[2*r]), zero);
        __m256i n0 = _mm256_sad_epu8(_mm256_and_si256(u0, w_neg[2*r]), zero);
        __m256i p1 = _mm256_sad_epu8(_mm256_and_si256(u1, w_pos[2*r+1]), zero);
        __m256i n1 = _mm256_sad_epu8(_mm256_and_si256(u1, w_neg[2*r+1]), zero);
        acc[r] = _mm256_add_epi64(_mm256_sub_epi64(p0, n0), _mm256_sub_epi64(p1, n1));
    }
    for (int r = 0; r < 4; r++)
        y[r] = (int32_t)(hsum_epi64_avx2(acc[r]) - c_row[r]);
}

/* ---------- 4. GEMV v2.1 bitplane online (2 bits/poids) ---------- */
static void htc_gemv_v21_bitplane_online_4x64(const uint8_t *u,
                                              const uint32_t *packed_pos,
                                              const uint32_t *packed_neg,
                                              const int32_t *c_row, int32_t *y) {
    __m256i u0 = _mm256_loadu_si256((const __m256i *)(u));
    __m256i u1 = _mm256_loadu_si256((const __m256i *)(u + 32));
    const __m256i zero = _mm256_setzero_si256();
    for (int r = 0; r < 4; r++) {
        __m256i wp0 = htc_expand_mask32_pure_avx2(packed_pos[2*r]);
        __m256i wp1 = htc_expand_mask32_pure_avx2(packed_pos[2*r+1]);
        __m256i wn0 = htc_expand_mask32_pure_avx2(packed_neg[2*r]);
        __m256i wn1 = htc_expand_mask32_pure_avx2(packed_neg[2*r+1]);
        __m256i p0 = _mm256_sad_epu8(_mm256_and_si256(u0, wp0), zero);
        __m256i n0 = _mm256_sad_epu8(_mm256_and_si256(u0, wn0), zero);
        __m256i p1 = _mm256_sad_epu8(_mm256_and_si256(u1, wp1), zero);
        __m256i n1 = _mm256_sad_epu8(_mm256_and_si256(u1, wn1), zero);
        __m256i diff = _mm256_add_epi64(_mm256_sub_epi64(p0, n0),
                                        _mm256_sub_epi64(p1, n1));
        y[r] = (int32_t)(hsum_epi64_avx2(diff) - c_row[r]);
    }
}

/* ---------- u = a + 128 (bijection exacte, = a XOR 0x80) ---------- */
static void shift128(const int8_t *a, uint8_t *u) {
    for (int k = 0; k < DIM; ++k) u[k] = (uint8_t)((int32_t)a[k] + 128);
}

/* ---------- Données partagées ---------- */
#define NROWS 1000
static int8_t Ws[NROWS * DIM], As[NROWS * DIM];
static uint8_t Us[NROWS * DIM], X127[NROWS * DIM];
static htc20_row_t R20[NROWS];
static htc21_row_t R21[NROWS];

static void init_data(void) {
    for (int i = 0; i < NROWS; ++i) {
        for (int k = 0; k < DIM; ++k) {
            Ws[i * DIM + k] = (int8_t)(rng_below(3)) - 1;
            As[i * DIM + k] = (int8_t)((int32_t)rng_below(256) - 128);
        }
        shift128(As + i * DIM, Us + i * DIM);
        for (int k = 0; k < DIM; ++k) {
            int32_t av = As[i * DIM + k];
            if (av < -127) av = -127;               /* clamping v2.0 */
            X127[i * DIM + k] = (uint8_t)(av + 127);
        }
        htc20_pack(Ws + i * DIM, &R20[i]);
        htc21_pack(Ws + i * DIM, &R21[i]);
    }
}

#ifndef HTC_FLYSIZE
int main(void) {
    printf("================================================================================\n");
    printf("  HTC-CORE v2.1 : VERIFICATION + BENCHMARK (gcc -O3 -march=native)\n");
    printf("================================================================================\n");

    /* ---------- 5. Vérification : le défaut -128 doit avoir disparu ---------- */
    const int32_t TRIALS = 500000;
    int32_t v20_fail = 0, v21_pre_fail = 0, v21_bp_fail = 0;
    int8_t w[DIM], a[DIM], w4[4 * DIM];
    uint8_t u[DIM], x127[DIM];
    htc20_row_t r20;
    htc21_row_t rows[4];
    __m256i wps[8], wns[8];
    uint32_t pp2[8], pn2[8];
    int32_t cs[4], y4[4];

    for (int32_t t = 0; t < TRIALS; ++t) {
        int boundary = (t < 5);
        for (int k = 0; k < DIM; ++k) {
            w[k] = (int8_t)(rng_below(3)) - 1;
            a[k] = (boundary && k == 0) ? -128 : (int8_t)((int32_t)rng_below(256) - 128);
        }
        shift128(a, u);
        for (int k = 0; k < DIM; ++k) {
            int32_t av = a[k];
            if (av < -127) av = -127;
            x127[k] = (uint8_t)(av + 127);
        }
        int32_t ref = ref_dot(w, a);

        htc20_pack(w, &r20);
        if (htc20_compute(&r20, x127) != ref) v20_fail++;

        for (int r = 0; r < 4; ++r) memcpy(w4 + r * DIM, w, DIM);
        for (int r = 0; r < 4; ++r) htc21_pack(w4 + r * DIM, &rows[r]);
        for (int r = 0; r < 4; ++r) {
            wps[2*r] = rows[r].wp[0]; wps[2*r+1] = rows[r].wp[1];
            wns[2*r] = rows[r].wn[0]; wns[2*r+1] = rows[r].wn[1];
            cs[r] = rows[r].c_row;
            pp2[2*r] = rows[r].pp[0]; pp2[2*r+1] = rows[r].pp[1];
            pn2[2*r] = rows[r].pn[0]; pn2[2*r+1] = rows[r].pn[1];
        }
        htc_gemv_v21_4x64(u, wps, wns, cs, y4);
        if (y4[0] != ref) v21_pre_fail++;
        htc_gemv_v21_bitplane_online_4x64(u, pp2, pn2, cs, y4);
        if (y4[0] != ref) v21_bp_fail++;
    }

    printf("\n--- Vérification (%d vecteurs, a ∈ [-128, 127]) ---\n", TRIALS);
    printf("  HTC v2.0 (offset 127 + clamping)          : %6d échecs%s\n",
           v20_fail, v20_fail ? " (défaut -128, attendu)" : "");
    printf("  HTC v2.1 GEMV pré-décompacté (offset 128) : %6d échecs\n", v21_pre_fail);
    printf("  HTC v2.1 GEMV bitplane online (offset128) : %6d échecs\n", v21_bp_fail);
    printf("  Cas limite a[0] = -128 inclus dans les 5 premiers essais.\n");

    /* ---------- 6. Benchmarks ---------- */
    init_data();
    const int ITERS = 500000, REPS = 5;
    volatile int32_t sink = 0;
    int32_t C4[4];
    double t_ref = 1e18, t_v20 = 1e18, t_pre = 1e18, t_bp = 1e18;

    for (int rep = 0; rep < REPS; ++rep) {
        double t0 = now_ns();
        for (int i = 0; i < ITERS; ++i)
            sink += ref_dot(Ws + (i % NROWS) * DIM, As + (i % NROWS) * DIM);
        double dt = (now_ns() - t0) / ITERS;
        if (dt < t_ref) t_ref = dt;

        t0 = now_ns();
        for (int i = 0; i < ITERS; ++i)
            sink += htc20_compute(&R20[i % NROWS], X127 + (i % NROWS) * DIM);
        dt = (now_ns() - t0) / ITERS;
        if (dt < t_v20) t_v20 = dt;

        t0 = now_ns();
        for (int i = 0; i < ITERS / 4; ++i) {
            int j = (i * 4) % NROWS;
            for (int r = 0; r < 4; ++r) C4[r] = R21[j + r].c_row;
            htc_gemv_v21_4x64(Us + j * DIM, (const __m256i *)&R21[j].wp[0],
                              (const __m256i *)&R21[j].wn[0], C4, y4);
            sink += y4[0] + y4[1] + y4[2] + y4[3];
        }
        dt = (now_ns() - t0) / (ITERS / 4) / 4.0;   /* normalisé par rangée */
        if (dt < t_pre) t_pre = dt;

        t0 = now_ns();
        for (int i = 0; i < ITERS / 4; ++i) {
            int j = (i * 4) % NROWS;
            for (int r = 0; r < 4; ++r) C4[r] = R21[j + r].c_row;
            htc_gemv_v21_bitplane_online_4x64(Us + j * DIM,
                                              (const uint32_t *)&R21[j].pp[0],
                                              (const uint32_t *)&R21[j].pn[0],
                                              C4, y4);
            sink += y4[0] + y4[1] + y4[2] + y4[3];
        }
        dt = (now_ns() - t0) / (ITERS / 4) / 4.0;
        if (dt < t_bp) t_bp = dt;
    }

    printf("\n--- Benchmark (ns / 64 MACs, min sur %d × %d appels) ---\n", REPS, ITERS);
    printf("  réf. int8 auto-vecto (gcc)     : %8.2f ns  (~%.1f GMACs/s)\n",
           t_ref, 64.0 / t_ref);
    printf("  HTC v2.0 scalaire (off.127)    : %8.2f ns  (~%.1f GMACs/s)\n",
           t_v20, 64.0 / t_v20);
    printf("  HTC v2.1 pré-décompacté AVX2   : %8.2f ns  (~%.1f GMACs/s)\n",
           t_pre, 64.0 / t_pre);
    printf("  HTC v2.1 bitplane online AVX2  : %8.2f ns  (~%.1f GMACs/s)\n",
           t_bp, 64.0 / t_bp);
    printf("  (sink=%d anti-élimination)\n", sink);
    return 0;
}
#endif /* HTC_FLYSIZE */
