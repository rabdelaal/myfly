/**
 * ============================================================================
 * htc_core.c — Portage du kernel HTC-Core v2.1 (GEMV ternaire AVX2) :
 * translation bijective +128 (a ^ 0x80, ZÉRO clamping, exact en -128),
 * accumulateur signé unique, décompression Bitplane Planaire pure AVX2.
 *
 * Contenu : htc_expand_mask32_pure_avx2, hsum_epi64_avx2,
 * htc_gemv_v21_bitplane_online_4x64 (poids 2 bits/poids),
 * htc_gemv_v21_preexp_4x64 (masques pré-étendus, même maths).
 * Référence scalaire int64 + harness : aléatoire + cas limites exhaustifs
 * (a_i dans {-128, -1, 0, 1, 127}) — le cas -128 est LE test du papier.
 *
 * Usage projet : primitive GEMV ternaire vérifiée, prête pour l'inférence
 * du readout quantifié (BitNet-style) quand le teacher arrivera. Les poids
 * MaleCNS ne sont PAS ternaires (cf. lif_alpha_ed.c) : ce kernel ne les
 * concerne pas — il concerne les futurs readouts {-1,0,+1}.
 *
 * Compilation :
 *   gcc -O3 -mavx2 -shared -o htc_core.dll htc_core.c
 *   gcc -O3 -mavx2 -o htc_verify.exe htc_core.c -DHTC_VERIFY_MAIN
 *   ./htc_verify.exe [bench]   (bench = timing, 10M itérations)
 * ============================================================================
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* Expansion pure AVX2 : 32 bits -> 32 octets de masque (0xFF/0x00), in-lane. */
static inline __m256i htc_expand_mask32_pure_avx2(uint32_t mask32) {
    __m256i v = _mm256_set1_epi32((int)mask32);
    const __m256i shuf = _mm256_setr_epi8(
        0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1,
        2,2,2,2, 2,2,2,2, 3,3,3,3, 3,3,3,3
    );
    __m256i bytes = _mm256_shuffle_epi8(v, shuf);
    const __m256i bit_mask = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128,
        1, 2, 4, 8, 16, 32, 64, (char)128
    );
    __m256i tested = _mm256_and_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(tested, bit_mask);
}

/* Réduction horizontale 4 x int64 -> int64. */
static inline int64_t hsum_epi64_avx2(__m256i v) {
    __m128i low   = _mm256_castsi256_si128(v);
    __m128i high  = _mm256_extracti128_si256(v, 1);
    __m128i sum   = _mm_add_epi64(low, high);
    __m128i hi64  = _mm_unpackhi_epi64(sum, sum);
    return _mm_cvtsi128_si64(_mm_add_epi64(sum, hi64));
}

/* Cœur partagé : u0/u1 déjà chargés, masques déjà étendus. */
static inline int32_t htc_row_accum(__m256i u0, __m256i u1,
                                    const __m256i *wp, const __m256i *wn,
                                    const __m256i zero, int32_t c_row) {
    __m256i p0 = _mm256_sad_epu8(_mm256_and_si256(u0, wp[0]), zero);
    __m256i n0 = _mm256_sad_epu8(_mm256_and_si256(u0, wn[0]), zero);
    __m256i p1 = _mm256_sad_epu8(_mm256_and_si256(u1, wp[1]), zero);
    __m256i n1 = _mm256_sad_epu8(_mm256_and_si256(u1, wn[1]), zero);
    __m256i diff = _mm256_add_epi64(_mm256_sub_epi64(p0, n0),
                                    _mm256_sub_epi64(p1, n1));
    return (int32_t)(hsum_epi64_avx2(diff) - c_row);
}

/* GEMV 4x64, poids bitplane (2 x uint32 par ligne et par polarité). */
void htc_gemv_v21_bitplane_online_4x64(
    const uint8_t* restrict u,
    const uint32_t* restrict packed_pos,
    const uint32_t* restrict packed_neg,
    const int32_t* restrict c_row,
    int32_t* restrict y)
{
    __m256i u0 = _mm256_loadu_si256((const __m256i*)(u));
    __m256i u1 = _mm256_loadu_si256((const __m256i*)(u + 32));
    const __m256i zero = _mm256_setzero_si256();
    for (int r = 0; r < 4; r++) {
        __m256i wp[2] = { htc_expand_mask32_pure_avx2(packed_pos[2*r]),
                          htc_expand_mask32_pure_avx2(packed_pos[2*r + 1]) };
        __m256i wn[2] = { htc_expand_mask32_pure_avx2(packed_neg[2*r]),
                          htc_expand_mask32_pure_avx2(packed_neg[2*r + 1]) };
        y[r] = htc_row_accum(u0, u1, wp, wn, zero, c_row[r]);
    }
}

/* GEMV 4x64, masques pré-étendus (2 x __m256i par ligne et par polarité). */
void htc_gemv_v21_preexp_4x64(
    const uint8_t* restrict u,
    const __m256i* restrict w_pos,
    const __m256i* restrict w_neg,
    const int32_t* restrict c_row,
    int32_t* restrict y)
{
    __m256i u0 = _mm256_loadu_si256((const __m256i*)(u));
    __m256i u1 = _mm256_loadu_si256((const __m256i*)(u + 32));
    const __m256i zero = _mm256_setzero_si256();
    for (int r = 0; r < 4; r++) {
        y[r] = htc_row_accum(u0, u1, &w_pos[2*r], &w_neg[2*r], zero, c_row[r]);
    }
}

#ifdef HTC_VERIFY_MAIN

static uint64_t htc_rng = 0x123456789ABCDEFULL;
static uint64_t htc_next(void) {
    htc_rng ^= htc_rng << 13; htc_rng ^= htc_rng >> 7; htc_rng ^= htc_rng << 17;
    return htc_rng;
}

/* Référence scalaire : dot(a, w) + bias, a signé, w ternaire. */
static int64_t htc_ref(const int8_t *a, const int8_t *w, int64_t bias) {
    int64_t s = bias;
    for (int i = 0; i < 64; ++i) s += (int64_t)a[i] * (int64_t)w[i];
    return s;
}

int main(int argc, char **argv) {
    printf("HTC-CORE v2.1 : harness (aleatoire + cas limites -128..127)\n");
    int fails = 0, tested = 0;
    uint8_t u[64];
    int8_t a[64], w[64];
    uint32_t pp[8], pn[8];
    __m256i wp[8], wn[8];
    int32_t c_row[4], y[4], y2[4];
    const int8_t edge[] = {-128, -1, 0, 1, 127};

    for (int t = 0; t < 20000; ++t) {
        int mode = t % 4;  /* 0: aléatoire, 1: tout -128, 2: alterné, 3: bords */
        for (int i = 0; i < 64; ++i) {
            int8_t av;
            if (mode == 0) av = (int8_t)(htc_next() & 0xFF);
            else if (mode == 1) av = -128;
            else if (mode == 2) av = (i % 2) ? 127 : -128;
            else av = edge[htc_next() % 5];
            a[i] = av;
            u[i] = (uint8_t)(av ^ 0x80);       /* offset bijectif, SANS clamp */
            int r = (int)(htc_next() % 3);     /* poids ternaires */
            w[i] = (int8_t)(r - 1);
        }
        int8_t wr[4][64];
        int64_t Bias[4];
        for (int r = 0; r < 4; ++r) {
            Bias[r] = (int64_t)(htc_next() % 2001) - 1000;
            int npos = 0, nneg = 0;
            pp[2*r] = pp[2*r+1] = pn[2*r] = pn[2*r+1] = 0;
            for (int i = 0; i < 64; ++i) {
                int8_t wv = (int8_t)((int)(htc_next() % 3) - 1);  /* -1,0,+1 */
                wr[r][i] = wv;
                if (wv > 0) { npos++; pp[2*r + (i / 32)] |= (1u << (i % 32)); }
                else if (wv < 0) { nneg++; pn[2*r + (i / 32)] |= (1u << (i % 32)); }
            }
            c_row[r] = (int32_t)(((npos - nneg) << 7) - Bias[r]);
            wp[2*r] = htc_expand_mask32_pure_avx2(pp[2*r]);
            wp[2*r+1] = htc_expand_mask32_pure_avx2(pp[2*r+1]);
            wn[2*r] = htc_expand_mask32_pure_avx2(pn[2*r]);
            wn[2*r+1] = htc_expand_mask32_pure_avx2(pn[2*r+1]);
        }
        htc_gemv_v21_bitplane_online_4x64(u, pp, pn, c_row, y);
        htc_gemv_v21_preexp_4x64(u, wp, wn, c_row, y2);
        /* Vérifie les 4 lignes contre la ref scalaire */
        for (int r = 0; r < 4; ++r) {
            int64_t expect = htc_ref(a, wr[r], Bias[r]);
            tested += 2;
            if ((int64_t)y[r] != expect) {
                if (fails < 3) printf("DIVERGE bitplane t=%d r=%d: %d vs %lld\n",
                                      t, r, y[r], (long long)expect);
                fails++;
            }
            if ((int64_t)y2[r] != expect) {
                if (fails < 3) printf("DIVERGE preexp t=%d r=%d\n", t, r);
                fails++;
            }
        }
    }
    printf("essais : %d produits, divergences : %d\n", tested, fails);
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        /* 16 tuiles distinctes, cyclées : rien ne peut être hoisté hors
         * boucle (le bug classique du bench mono-tile), tout reste en L1. */
        static uint8_t UU[16][64];
        static uint32_t PP[16][8], PN[16][8];
        static int32_t CC[16][4], YY[16][4];
        for (int k = 0; k < 16; ++k) {
            for (int i = 0; i < 64; ++i) {
                int8_t av = (int8_t)(htc_next() & 0xFF);
                UU[k][i] = (uint8_t)(av ^ 0x80);
            }
            for (int r = 0; r < 4; ++r) {
                int npos = 0, nneg = 0;
                PP[k][2*r] = PP[k][2*r+1] = PN[k][2*r] = PN[k][2*r+1] = 0;
                for (int i = 0; i < 64; ++i) {
                    int8_t wv = (int8_t)((int)(htc_next() % 3) - 1);
                    if (wv > 0) { npos++; PP[k][2*r + (i/32)] |= (1u << (i%32)); }
                    else if (wv < 0) { nneg++; PN[k][2*r + (i/32)] |= (1u << (i%32)); }
                }
                CC[k][r] = (int32_t)((npos - nneg) << 7);
            }
        }
        static __m256i WP[16][8], WN[16][8];
        for (int k = 0; k < 16; ++k)
            for (int j = 0; j < 8; ++j) {
                WP[k][j] = htc_expand_mask32_pure_avx2(PP[k][j]);
                WN[k][j] = htc_expand_mask32_pure_avx2(PN[k][j]);
            }
        const int IT = 10000000;
        volatile int64_t sink = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < IT; ++i) {
            int k = i & 15;
            htc_gemv_v21_bitplane_online_4x64(UU[k], PP[k], PN[k], CC[k], YY[k]);
            sink += YY[k][i & 3];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("bitplane online : %.2f ns/bloc-64-MACs (sink=%lld)\n",
               s / IT * 1e9, (long long)sink);
        sink = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < IT; ++i) {
            int k = i & 15;
            htc_gemv_v21_preexp_4x64(UU[k], WP[k], WN[k], CC[k], YY[k]);
            sink += YY[k][i & 3];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("pre-expanded    : %.2f ns/bloc-64-MACs (sink=%lld)\n",
               s / IT * 1e9, (long long)sink);
    }
    printf(fails ? "RESULTAT : ECHEC\n" : "RESULTAT : OK\n");
    return fails != 0;
}
#endif /* HTC_VERIFY_MAIN */
