/**
 * ============================================================================
 * lif_alpha_ed.c — Pas LIF α-synapse (Shiu et al., Nature 2024) EVENT-DRIVEN
 *                  en layout CSC, pour le cerveau MaleCNS en régime prod
 *                  (batch_size = 1).
 *
 * Objectif : battre le `torch.sparse.mm(self.W, delayed)` de backend/brain.py
 * (~51-71 ms/step sur MaleCNS) en ne dispersant que les neurones
 * PRÉSYNAPTIQUES ACTIFS (~7 %), au lieu de parcourir les n² synapses.
 *
 * Sémantique IDENTIQUE à brain.py::_step_prepared (modèle alpha) :
 *   delayed = self._delay.pop(0)            # spike reçu, retardé de delay_steps
 *   inc     = W @ delayed                   # dispersé seulement sur actifs
 *   g      *= syn_decay ; g += inc          # décroissance τ_syn
 *   V      += (-(V - V_rest) + neuromod * g) * (dt / tau_m)
 *   sp      = (V >= V_th) & (refr <= 0)
 *   V[sp]   = V_reset
 *   refr   -= 1, clamp>=0 ; refr[sp] = refractory_steps
 *   spikes_out = sp (float 0/1)
 *
 * Le buffer de retard (`_delay`) reste côté appelant : il passe le vecteur
 * `delayed` (n) de 0/1 déjà décodé par le pas précédent. Ce noyau ne
 * s'occupe que de l'intégration α + LIF + réfractaire.
 *
 * Layouts CSC (batch=1, tout en float32 / int32) :
 *   colptr (n+1) : colonne j = neurone présynaptique j
 *   row    (nnz) : neurone postsynaptique pour chaque arête
 *   data   (nnz) : poids w (mV)
 *   spikes (n)   : vecteur `delayed` 0/1 entrant
 *   V, g, refr (n) : état, modifié en place
 *   isensory (n) : booléen (0/1) neurone sensoriel (pour I_sensory en fq)
 *   I_sensory (n) : courant externe ; converti en Poisson par l'appelant
 *                   AVANT l'appel (codage fréquence) : seuls les sensoriels
 *                   actifs (Poisson) sont injectés en spikes par l'appelant.
 *                   → ici on ne gère QUE la part synaptique, les spikes
 *                     sensoriels arrivent déjà dans `delayed` via la boucle
 *                     sensory en amont.
 *
 * ponytail: les poids MaleCNS sont des ENTIERS EXACTS (100% a <=0.05 d'un
 * entier, range [-2591, 1878]) — on peut les stocker en int16 (2 octets) au
 * lieu de float32 (4 octets) : -2x de trafic memoire sur le scatter actif,
 * sans perte (conversion int16->float32 exacte pour ce range). PAS ternaire
 * (seuls 40% dans {-1,0,+1}) : le bitplane HTC-Core 2 bits perdrait 60% des
 * poids. A faire apres le benchmark de reference event-driven.
 *
 * Compilation :
 *   gcc -O3 -march=native -shared -o lif_alpha_ed.dll lif_alpha_ed.c
 *   gcc -O3 -march=native -o lif_alpha_ed_verify lif_alpha_ed.c -DLIF_VERIFY_MAIN
 * ============================================================================
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Un pas α-LIF event-driven, batch=1.
 * n = nombre de neurones, nnz = nombre d'arêtes.
 * colptr, row, data : CSC.
 * spikes  : entrée (n) 0/1 (déjà retardé + sensoriels Poisson).
 * V, g, refr : état, modifié en place.
 * V_rest, V_th, V_reset, neuromod, tau_m, dt, syn_decay, refractory_steps.
 * spikes_out : sortie (n) 0/1.
 */
void lif_alpha_ed_step(const int32_t *colptr, const int32_t *row, const float *data,
                       const int32_t n, const int32_t nnz,
                       const float *spikes, float *V, float *g, int32_t *refr,
                       const float V_rest, const float V_th, const float V_reset,
                       const float neuromod, const float tau_m, const float dt,
                       const float syn_decay, const int32_t refractory_steps,
                       float *spikes_out)
{
    /* --- Pass 1 : g *= syn_decay  puis  g[post] += w pour chaque actif --- */
    for (int32_t i = 0; i < n; ++i) g[i] *= syn_decay;

    for (int32_t j = 0; j < n; ++j) {
        if (spikes[j] == 0.0f) continue;          /* event-driven : skip ~93% */
        const int32_t lo = colptr[j], hi = colptr[j + 1];
        for (int32_t k = lo; k < hi; ++k) {
            g[row[k]] += data[k];                  /* + w * 1.0 */
        }
    }

    /* --- Pass 2 : intégration LIF + réfractaire absolu --- */
    const float integ = dt / tau_m;
    for (int32_t i = 0; i < n; ++i) {
        /* V += (-(V - V_rest) + neuromod * g) * integ */
        V[i] += (-(V[i] - V_rest) + neuromod * g[i]) * integ;

        const int32_t r = refr[i];
        const int spiked = (int32_t)(V[i] >= V_th) & (r <= 0);
        if (spiked) {
            V[i] = V_reset;
            refr[i] = refractory_steps;
            spikes_out[i] = 1.0f;
        } else {
            refr[i] = r > 0 ? r - 1 : 0;
            spikes_out[i] = 0.0f;
        }
    }
}

/**
 * Variante int16 : les poids MaleCNS sont des entiers exacts (100% a <=0.05
 * d'un entier, range [-2591, 1878]). Les stocker en int16 (2 octets) au lieu
 * de float32 (4 octets) divise par 2 le trafic memoire sur le scatter actif,
 * sans perte (int16->float32 exact pour ce range). Meme semantique que
 * lif_alpha_ed_step.
 */
void lif_alpha_ed_step_i16(const int32_t *colptr, const int32_t *row, const int16_t *data,
                           const int32_t n, const int32_t nnz,
                           const float *spikes, float *V, float *g, int32_t *refr,
                           const float V_rest, const float V_th, const float V_reset,
                           const float neuromod, const float tau_m, const float dt,
                           const float syn_decay, const int32_t refractory_steps,
                           float *spikes_out)
{
    for (int32_t i = 0; i < n; ++i) g[i] *= syn_decay;

    for (int32_t j = 0; j < n; ++j) {
        if (spikes[j] == 0.0f) continue;
        const int32_t lo = colptr[j], hi = colptr[j + 1];
        for (int32_t k = lo; k < hi; ++k) {
            g[row[k]] += (float)data[k];           /* int16->float32, exact */
        }
    }

    const float integ = dt / tau_m;
    for (int32_t i = 0; i < n; ++i) {
        V[i] += (-(V[i] - V_rest) + neuromod * g[i]) * integ;
        const int32_t r = refr[i];
        const int spiked = (int32_t)(V[i] >= V_th) & (r <= 0);
        if (spiked) {
            V[i] = V_reset;
            refr[i] = refractory_steps;
            spikes_out[i] = 1.0f;
        } else {
            refr[i] = r > 0 ? r - 1 : 0;
            spikes_out[i] = 0.0f;
        }
    }
}

/* Référence naïve — même accumulation (colonne active ordonnée j croissant),
 * bit-exact exigé avec lif_alpha_ed_step. */
void lif_alpha_ed_reference(const int32_t *colptr, const int32_t *row, const float *data,
                            const int32_t n, const int32_t nnz,
                            const float *spikes, float *V, float *g, int32_t *refr,
                            const float V_rest, const float V_th, const float V_reset,
                            const float neuromod, const float tau_m, const float dt,
                            const float syn_decay, const int32_t refractory_steps,
                            float *spikes_out)
{
    for (int32_t i = 0; i < n; ++i) {
        float acc = g[i] * syn_decay;
        for (int32_t j = 0; j < n; ++j) {
            if (spikes[j] == 0.0f) continue;
            const int32_t lo = colptr[j], hi = colptr[j + 1];
            for (int32_t k = lo; k < hi; ++k) {
                if (row[k] == i) { acc += data[k]; break; }
            }
        }
        g[i] = acc;
    }
    const float integ = dt / tau_m;
    for (int32_t i = 0; i < n; ++i) {
        V[i] += (-(V[i] - V_rest) + neuromod * g[i]) * integ;
        const int32_t r = refr[i];
        const int spiked = (int32_t)(V[i] >= V_th) & (r <= 0);
        if (spiked) { V[i] = V_reset; refr[i] = refractory_steps; spikes_out[i] = 1.0f; }
        else { refr[i] = r > 0 ? r - 1 : 0; spikes_out[i] = 0.0f; }
    }
}

#ifdef LIF_VERIFY_MAIN

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng_next(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
static uint32_t rng_below(uint32_t m) { return (uint32_t)(rng_next() % m); }
static float rng_uniform(void) { return (float)((rng_next() >> 40) & 0xFFFFFF) / (float)0xFFFFFF; }

int main(void) {
    printf("===============================================================================\n");
    printf("  LIF ALPHA EVENT-DRIVEN : HARNESS DE VERIFICATION\n");
    printf("===============================================================================\n");
#ifndef TRIALS
#define TRIALS 1000000
#endif
    const int32_t N_TRIALS = TRIALS;
    int32_t violations = 0, max_nnz = 0;

    float spikes[256], V[256], V_ref[256], g[256], g_ref[256], out[256], out_ref[256];
    int32_t refr[256], refr_ref[256];
    int32_t colptr[257], row[256 * 16];
    float data[256 * 16];

    for (int32_t t = 0; t < N_TRIALS; ++t) {
        const int32_t n = 1 + (int32_t)rng_below(256);
        int32_t nnz = 0;
        colptr[0] = 0;
        for (int32_t j = 0; j < n; ++j) {
            const int32_t deg = (int32_t)rng_below(16);
            int32_t prev = -1;
            for (int32_t d = 0; d < deg; ++d) {
                int32_t r = (int32_t)rng_below((uint32_t)n);
                if (r <= prev) r = prev + 1;
                if (r >= n) break;
                row[nnz] = r;
                data[nnz] = (rng_below(4) == 0) ? 0.0f : (rng_uniform() * 2.0f - 1.0f) * 0.1f;
                ++nnz; prev = r;
            }
            colptr[j + 1] = nnz;
        }
        if (nnz > max_nnz) max_nnz = nnz;

        const float V_rest = rng_uniform() * 0.5f;
        const float V_th   = 0.5f + rng_uniform();
        const float V_reset = (t % 7 == 0) ? -0.3f : V_rest;
        const float neuromod = (t % 5 == 0) ? 0.0f : 0.5f + rng_uniform() * 1.5f;
        const float tau_m = 1.0f + rng_uniform() * 39.0f;
        const float dt = (t % 11 == 0) ? tau_m : rng_uniform() * 5.0f;
        const float syn_decay = (t % 13 == 0) ? 1.0f : 0.1f + rng_uniform() * 0.89f;
        const int32_t refrac = 1 + (int32_t)rng_below(5);

        for (int32_t i = 0; i < n; ++i) {
            spikes[i] = (rng_below(4) == 0) ? 1.0f : 0.0f;
            V[i] = V_ref[i] = rng_uniform() * (V_th * 1.2f);
            g[i] = g_ref[i] = rng_uniform() * 0.8f;
            refr[i] = refr_ref[i] = (int32_t)rng_below(5);
        }

        float Vk[256]; memcpy(Vk, V, sizeof(float) * (size_t)n);
        float gk[256]; memcpy(gk, g, sizeof(float) * (size_t)n);
        int32_t rk[256]; memcpy(rk, refr, sizeof(int32_t) * (size_t)n);
        lif_alpha_ed_step(colptr, row, data, n, nnz, spikes, Vk, gk, rk,
                          V_rest, V_th, V_reset, neuromod, tau_m, dt, syn_decay, refrac, out);
        lif_alpha_ed_reference(colptr, row, data, n, nnz, spikes, V_ref, g_ref, refr_ref,
                               V_rest, V_th, V_reset, neuromod, tau_m, dt, syn_decay, refrac, out_ref);

        if (memcmp(Vk, V_ref, sizeof(float) * (size_t)n) != 0 ||
            memcmp(gk, g_ref, sizeof(float) * (size_t)n) != 0 ||
            memcmp(rk, refr_ref, sizeof(int32_t) * (size_t)n) != 0 ||
            memcmp(out, out_ref, sizeof(float) * (size_t)n) != 0) {
            if (violations < 5) fprintf(stderr, "DIVERGENCE trial=%d n=%d nnz=%d\n", t, n, nnz);
            violations++;
        }

        /* --- Variante int16 : les poids sont tronques a entiers (MaleCNS
         * est deja entier). La reference doit utiliser les memes poids
         * entiers retronques en float, sinon 0.07 != (int)0. --- */
        static int16_t data16[256 * 16];
        static float data_int[256 * 16];
        for (int32_t k = 0; k < nnz; ++k) {
            data16[k] = (int16_t)data[k];
            data_int[k] = (float)data16[k];   /* poids entier retronque */
        }
        float V16[256]; memcpy(V16, V, sizeof(float) * (size_t)n);
        float g16[256]; memcpy(g16, g, sizeof(float) * (size_t)n);
        int32_t r16[256]; memcpy(r16, refr, sizeof(int32_t) * (size_t)n);
        float out16[256], out16r[256];
        float V16r[256]; memcpy(V16r, V, sizeof(float) * (size_t)n);
        float g16r[256]; memcpy(g16r, g, sizeof(float) * (size_t)n);
        int32_t r16r[256]; memcpy(r16r, refr, sizeof(int32_t) * (size_t)n);
        lif_alpha_ed_step_i16(colptr, row, data16, n, nnz, spikes, V16, g16, r16,
                              V_rest, V_th, V_reset, neuromod, tau_m, dt, syn_decay, refrac, out16);
        lif_alpha_ed_reference(colptr, row, data_int, n, nnz, spikes, V16r, g16r, r16r,
                               V_rest, V_th, V_reset, neuromod, tau_m, dt, syn_decay, refrac, out16r);
        if (memcmp(V16, V16r, sizeof(float) * (size_t)n) != 0 ||
            memcmp(g16, g16r, sizeof(float) * (size_t)n) != 0 ||
            memcmp(r16, r16r, sizeof(int32_t) * (size_t)n) != 0 ||
            memcmp(out16, out16r, sizeof(float) * (size_t)n) != 0) {
            if (violations < 5) fprintf(stderr, "DIVERGENCE-i16 trial=%d n=%d nnz=%d\n", t, n, nnz);
            violations++;
        }
    }
    printf("Essais (N = %d), nnz max = %d :\n", N_TRIALS, max_nnz);
    printf("  - Bit-exact kernel vs reference : %d / %d (%.4f%%)\n",
           N_TRIALS - violations, N_TRIALS, 100.0 * (N_TRIALS - violations) / N_TRIALS);
    printf("  - Violations : %d\n", violations);
    return violations != 0;
}
#endif /* LIF_VERIFY_MAIN */