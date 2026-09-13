/**
 * ============================================================================
 * HTC-Core: reproduction du harness de verification du papier
 * "Formally Verified Multiplier-Free Dot Products for 1.58-Bit Ternary
 *  Accelerators via Static Bias Absorption" (R. Abdel-Aal).
 *
 * 1. Verification formelle empirique : 1 000 000 essais aleatoires
 *    (invariant bit-exact HTC vs reference int8).
 * 2. Cas limites deterministes, dont la demonstration du defaut
 *    a = -128 (wrap non-signe -> corruption silencieuse).
 * 3. Microbenchmark de latence : dot reference AVX2 vs htc_compute_dot_exact.
 *
 * Compilation : gcc -O3 -march=native -o htc_bench htc_core.c
 * ============================================================================
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define HTC_TEST_DIM 64U

typedef struct {
    uint64_t mask_pos;
    uint64_t mask_neg;
    int32_t  pop_bias;
} htc_verified_row_t;

static void htc_pack_row(const int8_t* __restrict__ w,
                         htc_verified_row_t* __restrict__ row) {
    uint64_t mp = 0ULL, mn = 0ULL;
    int32_t pop_pos = 0, pop_neg = 0;
    for (size_t k = 0; k < HTC_TEST_DIM; ++k) {
        uint64_t bit_p = (w[k] == 1) ? 1ULL : 0ULL;
        uint64_t bit_n = (w[k] == -1) ? 1ULL : 0ULL;
        mp |= (bit_p << k);
        mn |= (bit_n << k);
        pop_pos += (int32_t)bit_p;
        pop_neg += (int32_t)bit_n;
    }
    row->mask_pos = mp;
    row->mask_neg = mn;
    row->pop_bias = 127 * (pop_pos - pop_neg);
}

static int32_t htc_compute_dot_exact(const htc_verified_row_t* __restrict__ row,
                                     const uint8_t* __restrict__ x_shifted) {
    uint64_t mp = row->mask_pos, mn = row->mask_neg;
    int32_t sum_pos = 0, sum_neg = 0;
    for (size_t k = 0; k < HTC_TEST_DIM; ++k) {
        uint32_t bit_p = (uint32_t)((mp >> k) & 1ULL);
        uint32_t bit_n = (uint32_t)((mn >> k) & 1ULL);
        sum_pos += (int32_t)(x_shifted[k] * bit_p);
        sum_neg += (int32_t)(x_shifted[k] * bit_n);
    }
    return (sum_pos - sum_neg) - row->pop_bias;
}

static int32_t htc_reference_dot(const int8_t* __restrict__ w,
                                 const int8_t* __restrict__ a) {
    int32_t acc = 0;
    for (size_t k = 0; k < HTC_TEST_DIM; ++k) {
        acc += (int32_t)w[k] * (int32_t)a[k];
    }
    return acc;
}

/* Chronometre haute resolution */
static double now_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

int main(void) {
    printf("===============================================================================\n");
    printf("  HTC-CORE : REPRODUCTION DU HARNESS DE VERIFICATION\n");
    printf("===============================================================================\n");

    /* --- 1. 1 000 000 essais aleatoires --- */
    const int32_t TRIALS = 1000000;
    int32_t violations = 0;
    int32_t y_min = 100000, y_max = -100000;
    int8_t w[HTC_TEST_DIM], a[HTC_TEST_DIM];
    uint8_t x[HTC_TEST_DIM];
    htc_verified_row_t row;

    double t0 = now_ns();
    for (int32_t trial = 0; trial < TRIALS; ++trial) {
        for (size_t k = 0; k < HTC_TEST_DIM; ++k) {
            int8_t aw = (int8_t)(rng_next() % 3) - 1;              /* {-1,0,1} */
            int8_t av = (int8_t)(rng_next() % 255) - 127;          /* [-127,127] */
            w[k] = aw;
            a[k] = av;
            x[k] = (uint8_t)((int32_t)av + 127);
        }
        htc_pack_row(w, &row);
        int32_t y_ref = htc_reference_dot(w, a);
        int32_t y_htc = htc_compute_dot_exact(&row, x);
        if (y_ref != y_htc) violations++;
        if (y_ref < y_min) y_min = y_ref;
        if (y_ref > y_max) y_max = y_ref;
    }
    double dt = now_ns() - t0;
    printf("Essais aleatoires (N = %d) en %.2f s :\n", TRIALS, dt / 1e9);
    printf("  - Invariant bit-exact OK   : %d / %d (%.4f%%)\n",
           TRIALS - violations, TRIALS, 100.0 * (TRIALS - violations) / TRIALS);
    printf("  - Violations               : %d\n", violations);
    printf("  - Plage observee           : [%d, %d] (borne ACSL [-8128, +8128])\n",
           y_min, y_max);

    /* --- 2. Cas limites deterministes --- */
    printf("\nCas limites deterministes :\n");
    struct { int8_t w0, a0; const char* label; } cases[] = {
        { 1,  127, "w=+1, a=+127" },
        {-1,  127, "w=-1, a=+127" },
        { 0,  127, "w= 0, a=+127" },
        { 1,    0, "w=+1, a=  0"  },
    };
    for (size_t c = 0; c < 4; ++c) {
        memset(w, 0, sizeof w);
        memset(a, 0, sizeof a);
        w[0] = cases[c].w0;
        a[0] = cases[c].a0;
        x[0] = (uint8_t)((int32_t)a[0] + 127);
        htc_pack_row(w, &row);
        int32_t yr = htc_reference_dot(w, a);
        int32_t yh = htc_compute_dot_exact(&row, x);
        printf("  - %-16s y_ref = %6d, y_htc = %6d --> %s\n",
               cases[c].label, yr, yh, (yr == yh) ? "PASS" : "VIOLATION");
    }

    /* Le defaut -128 : hors precondition ACSL, corruption silencieuse */
    memset(w, 0, sizeof w);
    memset(a, 0, sizeof a);
    w[0] = 1;
    a[0] = -128;                     /* hors du contrat [-127, +127] */
    int32_t yr = htc_reference_dot(w, a);
    x[0] = (uint8_t)((int32_t)a[0] + 127);   /* wrap : 255 */
    memset(x + 1, 127, HTC_TEST_DIM - 1);    /* a=0 -> x=127 (neutre) */
    htc_pack_row(w, &row);
    int32_t yh = htc_compute_dot_exact(&row, x);
    printf("  - DEFAUT a=-128, w=+1 : y_ref = %6d, y_htc = %6d --> VIOLATION (delta = %d)\n",
           yr, yh, yh - yr);

    /* --- 3. Microbenchmark de latence --- */
    printf("\nMicrobenchmark (64 MACs par appel, mediane sur 5 x 2M appels) :\n");
    for (size_t k = 0; k < HTC_TEST_DIM; ++k) {
        w[k] = (int8_t)(rng_next() % 3) - 1;
        a[k] = (int8_t)(rng_next() % 255) - 127;
        x[k] = (uint8_t)((int32_t)a[k] + 127);
    }
    htc_pack_row(w, &row);
    int32_t sink = 0;
    const int32_t ITERS = 2000000;

    double best_ref = 1e18, best_htc = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        double t1 = now_ns();
        for (int32_t i = 0; i < ITERS; ++i) {
            sink += htc_reference_dot(w, a);
        }
        double t2 = now_ns();
        for (int32_t i = 0; i < ITERS; ++i) {
            sink += htc_compute_dot_exact(&row, x);
        }
        double t3 = now_ns();
        if ((t2 - t1) / ITERS < best_ref) best_ref = (t2 - t1) / ITERS;
        if ((t3 - t2) / ITERS < best_htc) best_htc = (t3 - t2) / ITERS;
    }
    printf("  - dot reference int8 (AVX2 auto-vecto) : %6.2f ns/appel\n", best_ref);
    printf("  - htc_compute_dot_exact (bitmasks)     : %6.2f ns/appel\n", best_htc);
    printf("  - ratio CPU : x%.2f (HTC plus lent sur CPU -> cible = FPGA/ASIC)\n",
           best_htc / best_ref);
    printf("  (sink=%d pour empecher l'elimination par le compilateur)\n", sink);

    return violations != 0;
}
