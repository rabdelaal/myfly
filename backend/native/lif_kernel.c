/**
 * ============================================================================
 * lif_kernel.c — Pas LIF sparse CSR batché, à la manière HTC-Core :
 * un noyau C autonome, vérifié par un harness à 1M vecteurs aléatoires
 * bit-exact contre une référence naïve, et conçu pour être appelé depuis
 * Python (ctypes) avec zéro allocation par pas.
 *
 * Sémantique identique à backend/brain.py::FlyBrain.step :
 *   I_syn[b,i] = neuromod * sum_k data[k] * spikes[b, indices[k]]  (+ I_full)
 *   V'         = V + (-V + I_syn) * dt/tau_m
 *   spiked     = V' >= V_th ;  V <- spiked ? V_reset : V' ;  spikes_out = spiked
 *
 * Layouts (row-major) :
 *   spikes, V, I_full, spikes_out : (n, B) — ligne i contiguë sur B
 *   indptr (n+1), indices (nnz), data (nnz) : CSR classique
 *   I_full : (n, B) ou NULL — courants externes déjà dispersés sur les
 *            neurones sensoriels par l'appelant.
 *
 * Harness (main) : 1 000 000 réseaux aléatoires n<=64 — le kernel et la
 * référence accumulent dans le MÊME ordre (k croissant), donc le bit-exact
 * est exigible ; toute divergence = bug d'indexation ou de débordement.
 *
 * Compilation : gcc -O3 -march=native -shared -o lif_kernel.dll lif_kernel.c
 *               gcc -O3 -march=native -o lif_verify lif_kernel.c -DHTC_VERIFY_MAIN
 * ============================================================================
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Un pas de temps pour tout le batch. Ne retourne rien : écrit dans
 * V, spikes_out. isyn est un buffer de travail (n*B) fourni par l'appelant.
 */
void lif_step(const int32_t *indptr, const int32_t *indices, const float *data,
              const int32_t n, const int32_t B,
              const float *spikes, float *V, float *isyn,
              const float *I_full,
              const float neuromod, const float tau_m, const float dt,
              const float V_th, const float V_reset,
              float *spikes_out)
{
    const float decay = dt / tau_m;

    /* Pass 1 : I_syn = W @ spikes, ligne i contiguë sur b (cache-friendly) */
    memset(isyn, 0, (size_t)n * B * sizeof(float));
    for (int32_t i = 0; i < n; ++i) {
        float *out_row = isyn + (size_t)i * B;
        const int32_t lo = indptr[i], hi = indptr[i + 1];
        for (int32_t k = lo; k < hi; ++k) {
            const float w = data[k];
            const float *srow = spikes + (size_t)indices[k] * B;
            for (int32_t b = 0; b < B; ++b) {
                out_row[b] += w * srow[b];
            }
        }
    }

    /* Pass 2 : intégration LIF élément par élément */
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t b = 0; b < B; ++b) {
            const size_t idx = (size_t)i * B + b;
            float I = neuromod * isyn[idx];
            if (I_full != NULL) I += I_full[idx];
            float v = V[idx] + (-V[idx] + I) * decay;
            const uint8_t spiked = (uint8_t)(v >= V_th);
            V[idx] = spiked ? V_reset : v;
            spikes_out[idx] = (float)spiked;
        }
    }
}

/**
 * Référence naïve, même ordre d'accumulation (k croissant) par sortie —
 * bit-exact exigé avec lif_step.
 */
void lif_step_reference(const int32_t *indptr, const int32_t *indices, const float *data,
                        const int32_t n, const int32_t B,
                        const float *spikes, float *V,
                        const float *I_full,
                        const float neuromod, const float tau_m, const float dt,
                        const float V_th, const float V_reset,
                        float *spikes_out)
{
    const float decay = dt / tau_m;
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int32_t k = indptr[i]; k < indptr[i + 1]; ++k) {
                sum += data[k] * spikes[(size_t)indices[k] * B + b];
            }
            float I = neuromod * sum;
            if (I_full != NULL) I += I_full[(size_t)i * B + b];
            const size_t idx = (size_t)i * B + b;
            float v = V[idx] + (-V[idx] + I) * decay;
            const uint8_t spiked = (uint8_t)(v >= V_th);
            V[idx] = spiked ? V_reset : v;
            spikes_out[idx] = (float)spiked;
        }
    }
}

#ifdef HTC_VERIFY_MAIN

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static uint32_t rng_below(uint32_t m) { return (uint32_t)(rng_next() % m); }
static float rng_uniform(void) { return (float)((rng_next() >> 40) & 0xFFFFFF) / (float)0xFFFFFF; }

int main(void) {
    printf("===============================================================================\n");
    printf("  LIF KERNEL : HARNESS DE VERIFICATION (1 000 000 reseaux aleatoires)\n");
    printf("===============================================================================\n");

    const int32_t TRIALS = 1000000;
    int32_t violations = 0;
    int32_t max_nnz = 0;

    float spikes[64 * 8], V[64 * 8], V_ref[64 * 8], isyn[64 * 8];
    float I_full[64 * 8], out[64 * 8], out_ref[64 * 8];
    int32_t indptr[65], indices[64 * 8];
    float data[64 * 8];

    for (int32_t trial = 0; trial < TRIALS; ++trial) {
        const int32_t n = 1 + (int32_t)rng_below(64);
        const int32_t B = 1 + (int32_t)rng_below(8);

        /* CSR aléatoire, colonnes triées par rangée (invariant CSR) */
        int32_t nnz = 0;
        indptr[0] = 0;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t deg = (int32_t)rng_below(9); /* 0..8 synapses */
            int32_t prev = -1;
            for (int32_t d = 0; d < deg; ++d) {
                int32_t col = (int32_t)rng_below((uint32_t)n);
                if (col <= prev) col = prev + 1;
                if (col >= n) break;
                indices[nnz] = col;
                /* Poids symétriques, échelle variée, parfois nuls */
                data[nnz] = (rng_below(4) == 0) ? 0.0f
                            : (rng_uniform() * 2.0f - 1.0f) * (0.1f + rng_uniform());
                ++nnz;
                prev = col;
            }
            indptr[i + 1] = nnz;
        }
        if (nnz > max_nnz) max_nnz = nnz;

        const float neuromod = (trial % 7 == 0) ? 0.0f : 0.3f + rng_uniform() * 1.7f;
        const float tau_m    = 1.0f + rng_uniform() * 39.0f;
        const float dt       = (trial % 11 == 0) ? tau_m : rng_uniform() * 5.0f;
        const float V_th     = 0.5f + rng_uniform();
        const float V_reset  = (trial % 13 == 0) ? -0.2f : 0.0f;
        const int use_I      = (trial % 3 != 0);

        for (int32_t j = 0; j < n * B; ++j) {
            spikes[j] = (rng_below(4) == 0) ? 1.0f : 0.0f;
            V[j] = V_ref[j] = rng_uniform() * (V_th * 1.2f);
            I_full[j] = use_I ? (rng_uniform() * 3.0f - 1.5f) : 0.0f;
        }
        const float *I_ptr = use_I ? I_full : NULL;

        float V_k[64 * 8];
        memcpy(V_k, V, sizeof(float) * (size_t)n * B);
        lif_step(indptr, indices, data, n, B, spikes, V_k, isyn, I_ptr,
                 neuromod, tau_m, dt, V_th, V_reset, out);
        lif_step_reference(indptr, indices, data, n, B, spikes, V_ref, I_ptr,
                           neuromod, tau_m, dt, V_th, V_reset, out_ref);

        if (memcmp(V_k, V_ref, sizeof(float) * (size_t)n * B) != 0 ||
            memcmp(out, out_ref, sizeof(float) * (size_t)n * B) != 0) {
            if (violations < 5) {
                fprintf(stderr, "DIVERGENCE trial=%d n=%d B=%d\n", trial, n, B);
            }
            violations++;
        }
    }

    printf("Essais aleatoires (N = %d), nnz max = %d :\n", TRIALS, max_nnz);
    printf("  - Bit-exact kernel vs reference : %d / %d (%.4f%%)\n",
           TRIALS - violations, TRIALS, 100.0 * (TRIALS - violations) / TRIALS);
    printf("  - Violations : %d\n", violations);
    printf("  Cas couverts : neuromod=0, dt=tau (decay=1), V_reset negatif,\n");
    printf("  rangees vides, poids nuls, I_full absent.\n");
    return violations != 0;
}
#endif /* HTC_VERIFY_MAIN */
