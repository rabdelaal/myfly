/**
 * ============================================================================
 * sigil_circuit.c — Circuits virtuels dont la TOPOLOGIE vient des sceaux :
 * un sigil est un graphe (nœuds = croisements/terminaisons, arêtes = traits
 * et arcs, cercles = boucles récurrentes). On génère l'edge-list du pattern,
 * on l'instancie en réseau LIF, et on mesure ce qui ÉMERGE (synchronie,
 * fréquence dominante, capacité mémoire) + la vitesse (pas/s).
 *
 * Patterns (motifs de sceaux de Salomon & autres) :
 *   0 SEAL      : double anneau concentrique + rayons + croix diamétrale
 *                 (le sceau classique : cercles + croix inscrits)
 *   1 PENTAGRAM : anneau + cordes à saut k (étoile {n/k}, pentagramme)
 *   2 RING      : cycle orienté simple (cercle magique, onde circulante)
 *   3 WHEEL     : anneau + moyeu bidirectionnel (roue, sceau à centre)
 *   4 GRID      : grille torique (tablettes, carrés magiques)
 *   5 ZIGGURAT  : 4 couches feedforward + anneaux intra-couche (layer map)
 *   6 RANDOM    : contrôle Erdős–Rényi, même budget d'arêtes (témoin)
 *
 * Design : les arêtes sont orientées (src -> dst). RING est un cycle orienté
 * (onde qui tourne) ; les autres motifs sont bidirectionnels (réverbération).
 * Les POIDS (+/-) et la normalisation du rayon spectral se font côté appelant
 * (Python/numpy) : la topologie pure pilote les différences émergentes.
 *
 * Sémantique du pas (LIF courant, identique à lif_kernel.c) :
 *   I[i] = inp[i] + sum_j W[i*n+j] * spikes[j]
 *   V[i] += (-V[i] + I[i]) * decay ; spike si V >= V_th (reset sinon).
 *
 * Compilation :
 *   gcc -O3 -march=native -shared -o sigil_circuit.dll sigil_circuit.c
 *   gcc -O3 -march=native -o sigil_verify.exe sigil_circuit.c -DSIGIL_VERIFY_MAIN
 * ============================================================================
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SIGIL_SEAL 0
#define SIGIL_PENTAGRAM 1
#define SIGIL_RING 2
#define SIGIL_WHEEL 3
#define SIGIL_GRID 4
#define SIGIL_ZIGGURAT 5
#define SIGIL_RANDOM 6
#define SIGIL_NPATTERNS 7

static const char *SIGIL_NAMES[SIGIL_NPATTERNS] = {
    "seal", "pentagram", "ring", "wheel", "grid", "ziggurat", "random"
};

static uint32_t lcg_next(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static int push_edge(int32_t *src, int32_t *dst, int max_edges, int *m,
                     int32_t a, int32_t b) {
    if (*m >= max_edges || a == b) return 0;
    src[*m] = a; dst[*m] = b; (*m)++;
    return 1;
}

static void push2(int32_t *src, int32_t *dst, int max_edges, int *m,
                  int32_t a, int32_t b) {
    push_edge(src, dst, max_edges, m, a, b);
    push_edge(src, dst, max_edges, m, b, a);
}

/**
 * Génère l'edge-list du pattern. Retourne m (nombre d'arêtes).
 * Exige max_edges >= 8*n (marge pour tous les motifs).
 */
int sigil_edges(int pattern, int n, uint32_t seed,
                int32_t *src, int32_t *dst, int max_edges) {
    int m = 0;
    if (n < 4 || max_edges < 8 * n) return -1;
    uint32_t rng = seed ? seed : 0xC0FFEEu;

    if (pattern == SIGIL_RING) {
        for (int i = 0; i < n; ++i)
            push_edge(src, dst, max_edges, &m, i, (i + 1) % n);
    } else if (pattern == SIGIL_PENTAGRAM) {
        int k = n / 2 - 1;
        if (k < 2) k = 2;
        for (int i = 0; i < n; ++i) {
            push_edge(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + k) % n);
        }
    } else if (pattern == SIGIL_WHEEL) {
        for (int i = 0; i < n; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
        for (int j = 1; j < n; ++j)
            push2(src, dst, max_edges, &m, 0, j);
    } else if (pattern == SIGIL_SEAL) {
        int no = (2 * n) / 3;             /* anneau externe */
        if (no < 4) no = 4;
        int ni = n - no;                  /* anneau interne */
        if (ni < 2) { ni = 2; no = n - 2; }
        for (int i = 0; i < no; ++i)      /* cercles concentriques */
            push2(src, dst, max_edges, &m, i, (i + 1) % no);
        for (int i = 0; i < ni; ++i)
            push2(src, dst, max_edges, &m, no + i, no + ((i + 1) % ni));
        for (int i = 0; i < no; ++i)      /* rayons externe -> interne */
            push2(src, dst, max_edges, &m, i, no + (i % ni));
        push2(src, dst, max_edges, &m, 0, no / 2);          /* croix */
        push2(src, dst, max_edges, &m, no, no + ni / 2);
    } else if (pattern == SIGIL_GRID) {
        int s = 1;
        while ((s + 1) * (s + 1) <= n) ++s;   /* pas torique */
        if (s < 2) s = 2;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + s) % n);
        }
    } else if (pattern == SIGIL_ZIGGURAT) {
        const int L = 4, base_m = n / L;
        if (base_m < 1) return -1;
        for (int i = 0; i < n; ++i) {
            int l = (i * L) / n;                       /* couche de i */
            int b0 = (l * n) / L, b1 = ((l + 1) * n) / L;
            int sz = b1 - b0;                          /* taille couche */
            push2(src, dst, max_edges, &m, i, b0 + ((i - b0 + 1) % sz));
            if (l < L - 1) {                           /* feedforward x2 */
                int nb0 = ((l + 1) * n) / L, nb1 = ((l + 2) * n) / L;
                int nsz = nb1 - nb0, k = (i - b0) % sz;
                push_edge(src, dst, max_edges, &m, i, nb0 + ((2 * k) % nsz));
                push_edge(src, dst, max_edges, &m, i, nb0 + ((2 * k + 1) % nsz));
            }
        }
    } else if (pattern == SIGIL_RANDOM) {
        int target = 3 * n;
        int guard = 0;
        while (m < target && guard++ < target * 20) {
            int32_t a = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            int32_t b = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            push_edge(src, dst, max_edges, &m, a, b);
        }
    } else {
        return -1;
    }
    return m;
}

/** Un pas LIF dense, batch=1. W row-major (n*n). Buffers spikes_in/out
 *  séparés (pas d'aliasing) : V (n) modifié en place. */
void sigil_step(const float *W, const int n,
                const float *spikes_in, const float *inp, float *V,
                const float V_th, const float V_reset, const float decay,
                float *spikes_out) {
    for (int i = 0; i < n; ++i) {
        const float *row = W + (size_t)i * n;
        float acc = inp[i];
        for (int j = 0; j < n; ++j) acc += row[j] * spikes_in[j];
        float v = V[i] + (-V[i] + acc) * decay;
        const int sp = (int)(v >= V_th);
        V[i] = sp ? V_reset : v;
        spikes_out[i] = (float)sp;
    }
}

#ifdef SIGIL_VERIFY_MAIN
/* ... harness ci-dessous ... */
static void sigil_step_ref(const float *W, const int n,
                           const float *inp, const float *spikes, float *V,
                           const float V_th, const float V_reset,
                           const float decay, float *out) {
    for (int i = 0; i < n; ++i) {
        float acc = inp[i];
        for (int j = 0; j < n; ++j) acc += W[(size_t)i * n + j] * spikes[j];
        float v = V[i] + (-V[i] + acc) * decay;
        const int sp = (int)(v >= V_th);
        V[i] = sp ? V_reset : v;
        out[i] = (float)sp;
    }
}

int main(void) {
    printf("SIGIL_CIRCUIT : harness (invariants + bit-exact pas)\n");
    int fails = 0;
    int32_t src[8 * 128], dst[8 * 128];
    /* invariants par motif */
    {
        int m = sigil_edges(SIGIL_RING, 64, 1, src, dst, 8 * 128);
        if (m != 64) { printf("FAIL ring m=%d\n", m); fails++; }
        m = sigil_edges(SIGIL_SEAL, 64, 1, src, dst, 8 * 128);
        if (m <= 0) { printf("FAIL seal\n"); fails++; }
        for (int p = 0; p < SIGIL_NPATTERNS; ++p) {
            m = sigil_edges(p, 64, 1234, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL pattern %d\n", p); fails++; continue; }
            for (int k = 0; k < m; ++k)
                if (src[k] < 0 || src[k] >= 64 || dst[k] < 0 || dst[k] >= 64 ||
                    src[k] == dst[k]) {
                    printf("FAIL pattern %d edge %d\n", p, k); fails++; break;
                }
        }
        if (sigil_edges(99, 64, 1, src, dst, 8 * 128) != -1) { printf("FAIL bad pattern\n"); fails++; }
        if (sigil_edges(0, 2, 1, src, dst, 8 * 128) != -1) { printf("FAIL small n\n"); fails++; }
    }
    /* bit-exact kernel vs ref sur petits réseaux aléatoires */
    {
        uint32_t rng = 42;
        float W[32 * 32], V[32], Vr[32], inp[32], sp[32], out[32], outr[32];
        int bad = 0;
        for (int t = 0; t < 2000; ++t) {
            int n = 4 + (int)(lcg_next(&rng) % 29);
            for (int i = 0; i < n * n; ++i)
                W[i] = ((float)(lcg_next(&rng) % 200) / 100.0f - 1.0f) * 0.5f;
            for (int i = 0; i < n; ++i) {
                sp[i] = (float)(lcg_next(&rng) % 3 == 0);
                V[i] = Vr[i] = (float)(lcg_next(&rng) % 100) / 100.0f;
                inp[i] = (float)(lcg_next(&rng) % 100) / 100.0f - 0.5f;
            }
            float Vk[32];
            memcpy(Vk, V, sizeof(float) * (size_t)n);
            sigil_step(W, n, sp, inp, Vk, 1.0f, 0.0f, 0.05f, out);
            sigil_step_ref(W, n, inp, sp, Vr, 1.0f, 0.0f, 0.05f, outr);
            if (memcmp(Vk, Vr, sizeof(float) * (size_t)n) != 0 ||
                memcmp(out, outr, sizeof(float) * (size_t)n) != 0) {
                if (bad < 3) printf("DIVERGENCE t=%d n=%d\n", t, n);
                bad++;
            }
        }
        printf("bit-exact kernel vs ref : %d/2000 bad (attendu 0)\n", bad);
        fails += bad;
    }
    printf(fails ? "RESULTAT : ECHEC\n" : "RESULTAT : OK\n");
    return fails != 0;
}
#endif /* SIGIL_VERIFY_MAIN */
