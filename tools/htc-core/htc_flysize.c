/**
 * ============================================================================
 * htc_flysize.c — Test à l'échelle de la mouche :
 * GEMV ternaire DENSE bitplane HTC v2.1 (5056×5056, batch 20)
 * vs notre pas LIF torch sparse CSR (~4.5 ms mesurés dans bench_htc_fly.py).
 *
 * Layout bitplane : [groupe de 4 rangées][bloc de 64 cols][r][lo,hi]
 * -> 8 uint32 contigus par appel kernel, exactement ce que attend
 *    htc_gemv_v21_bitplane_online_4x64.
 *
 * Compilation : gcc -O3 -march=native -DHTC_FLYSIZE -o htc_flysize htc_flysize.c
 * ============================================================================
 */
#define HTC_FLYSIZE 1
#include "htc_v21.c"

#define N 5056                 /* 5000 neurones paddés à un multiple de 64 */
#define NBLK (N / 64)          /* 79 blocs de 64 colonnes */
#define NGROUP (N / 4)         /* 1264 groupes de 4 rangées */
#define BATCH 20

int main(void) {
    printf("================================================================================\n");
    printf("  FLY-SIZE : GEMV ternaire dense bitplane (%dx%d, batch %d) vs sparse CSR\n", N, N, BATCH);
    printf("================================================================================\n");

    /* chunks[g][blk] : 8 uint32 = 4 rangées × (lo,hi) pour pos puis neg */
    const size_t CHUNKS = (size_t)NGROUP * NBLK;
    uint32_t *pp = malloc(CHUNKS * 8 * sizeof(uint32_t));
    uint32_t *pn = malloc(CHUNKS * 8 * sizeof(uint32_t));
    int32_t *c_row = malloc((size_t)N * sizeof(int32_t));
    uint8_t *u = malloc((size_t)BATCH * N);
    int32_t *y = malloc((size_t)N * sizeof(int32_t));

    for (int g = 0; g < NGROUP; ++g) {
        for (int blk = 0; blk < NBLK; ++blk) {
            uint32_t *cp = pp + ((size_t)g * NBLK + blk) * 8;
            uint32_t *cn = pn + ((size_t)g * NBLK + blk) * 8;
            for (int r = 0; r < 4; ++r) {
                uint32_t lo_p = (uint32_t)rng_next(), hi_p = (uint32_t)rng_next();
                uint32_t lo_n = (uint32_t)rng_next(), hi_n = (uint32_t)rng_next();
                lo_n &= ~lo_p; hi_n &= ~hi_p;   /* ternaire : pos et neg exclusifs */
                cp[2*r] = lo_p; cp[2*r+1] = hi_p;
                cn[2*r] = lo_n; cn[2*r+1] = hi_n;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        int npos = 0, nneg = 0;
        const int g = i / 4, r = i % 4;
        for (int blk = 0; blk < NBLK; ++blk) {
            const uint32_t *cp = pp + ((size_t)g * NBLK + blk) * 8 + 2 * r;
            const uint32_t *cn = pn + ((size_t)g * NBLK + blk) * 8 + 2 * r;
            npos += __builtin_popcount(cp[0]) + __builtin_popcount(cp[1]);
            nneg += __builtin_popcount(cn[0]) + __builtin_popcount(cn[1]);
        }
        c_row[i] = 128 * (npos - nneg);
    }
    /* Activations : spikes ∈ {0,1} -> a ∈ {0,1} -> u = a+128 ∈ {128,129} */
    for (size_t j = 0; j < (size_t)BATCH * N; ++j) u[j] = 128 + (rng_below(4) == 0);

    printf("  Mémoire bitplanes (pos+neg) : %.2f Mo (sparse CSR réel : 0.91 Mo)\n",
           2.0 * CHUNKS * 8 * sizeof(uint32_t) / 1e6);

    volatile int32_t sink = 0;
    double best = 1e18;
    const int REPS = 5;
    for (int rep = 0; rep < REPS; ++rep) {
        double t0 = now_ns();
        for (int b = 0; b < BATCH; ++b) {
            const uint8_t *ub = u + (size_t)b * N;
            memset(y, 0, (size_t)N * sizeof(int32_t));
            for (int g = 0; g < NGROUP; ++g) {
                int32_t y4[4], C4[4];
                for (int r = 0; r < 4; ++r) C4[r] = c_row[4 * g + r];
                for (int blk = 0; blk < NBLK; ++blk) {
                    /* Accumuler les 79 blocs dans y4 via hsum : le kernel
                       écrit y[r] = somme - c ; on soustrait c_row une fois
                       par bloc ? Non : c_row est constant par rangée, on
                       l'applique donc en une seule passe finale. Simplifions :
                       on laisse le kernel faire bloc par bloc avec C4 nul,
                       puis on corrige après coup. */
                    int32_t C0[4] = {0, 0, 0, 0};
                    htc_gemv_v21_bitplane_online_4x64(
                        ub, pp + ((size_t)g * NBLK + blk) * 8,
                        pn + ((size_t)g * NBLK + blk) * 8, C0, y4);
                    for (int r = 0; r < 4; ++r) y[4 * g + r] += y4[r];
                }
            }
            for (int i = 0; i < N; ++i) y[i] -= c_row[i];
        }
        double dt = (now_ns() - t0) / 1e6;
        if (dt < best) best = dt;
        sink += y[0];
    }

    printf("  GEMV dense HTC v2.1 bitplane : %.2f ms (min %d reps, 20 colonnes)\n",
           best, REPS);
    printf("  Pas LIF torch sparse CSR     :    4.50 ms (mesuré, bench_htc_fly.py)\n");
    printf("  -> ratio : x%.1f en défaveur du dense (25.6M MACs vs 113k synapses)\n",
           best / 4.50);
    printf("  (sink=%d)\n", sink);
    free(pp); free(pn); free(c_row); free(u); free(y);
    (void)sink;
    return 0;
}
