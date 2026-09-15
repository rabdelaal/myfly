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
 *   7 HEXAGRAM  : Sceau de Salomon : deux triangles entrelacés (cordes n/3)
 *   8 PENTACLE  : cercle + étoile (anneau bidir + cordes d'étoile)
 *   9 TREE      : Arbre de Vie kabbalistique : 10 sephiroth + 22 sentiers
 *                 (taille canonique n=10, cf. sigil_native_n)
 *  10 OUROBOROS : cycle + corde de feedback lointaine (tête qui mord)
 *  11 TRIQUETRA : trois anneaux liés en triangle (celtique)
 *  12 ANKH      : anneau + barre diamétrale + queue (croix de vie)
 *  13 SRIYANTA  : 3 anneaux triangulaires + rayons inter-couches
 *  14 TRISKEL   : moyeu + 3 bras récurrents (celtique)
 *  15 VESICA    : deux anneaux partageant 2 nœuds (vesica piscis)
 *  16 ISA       : ligne ouverte orientée (rune : delay line pure)
 *  17 FEHU      : ligne + brindilles i->i+2 (richesse qui rejoint)
 *  18 ALGIZ     : ligne + feedbacks locaux (protection, rune de vie)
 *  19 HAGALAZ   : échelle : deux lignes + barreaux (grêle, H répété)
 *  20 OTHALA    : chaîne de losanges (héritage, rhombus tiling)
 *  21 BINDRUNE  : ligne + 8 cordes scellées par seed (rune liée)
 *  22 FLOWER    : Fleur de Vie : treillis hexagonal torique
 *  23 METATRON  : Cube de Métatron : graphe complet K13 (n=13)
 *  24 ICHING    : Yi Jing : 64 hexagrammes = hypercube Q6 (n=64)
 *  25 BAGUA     : Bagua : 8 trigrammes = cube Q3 (n=8)
 *  26 YETZIRAH  : Sefer Yetzirah : 231 portes = K22 complet (n=22)
 *  27 LABYRINTH : labyrinthe unicursal : chemin hamiltonien en serpent
 *  28 KOLAM     : pulli kolam : tore à 8-connectivité (boucles fermées)
 *  29 GOETIC    : sceau goétique : anneau + 8 cordes internes (seed)
 *  30 ENOCHIAN  : Sigillum Dei : anneau + heptagramme + moyeu
 *  31 VEVE      : vèvè : carrefour en croix + anneau + paires miroir
 *  32 OGHAM    : ogham : bâton + rameaux culs-de-sac (sinks)
 *  33 ADINKRA   : nkyinkyim : zigzag + torsades longues
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
/* --- Sceaux sacrés : Kabbale, Salomon, Égypte, Celtes, Inde --- */
#define SIGIL_HEXAGRAM 7    /* Sceau de Salomon : deux triangles entrelacés */
#define SIGIL_PENTACLE 8    /* cercle + étoile (anneau bidir + cordes) */
#define SIGIL_TREE 9        /* Arbre de Vie : 10 sephiroth + 22 sentiers (n=10) */
#define SIGIL_OUROBOROS 10  /* serpent qui se mord : cycle + feedback lointain */
#define SIGIL_TRIQUETRA 11  /* trois anneaux en triangle (celtique) */
#define SIGIL_ANKH 12       /* anneau + barre + queue (croix de vie) */
#define SIGIL_SRIYANTA 13   /* 3 anneaux triangulaires + rayons (Sri Yantra) */
#define SIGIL_TRISKEL 14    /* moyeu + 3 bras récurrents (celtique) */
#define SIGIL_VESICA 15     /* deux anneaux partageant 2 nœuds (vesica piscis) */
/* --- Runes nordiques : famille morphologique (lignes, branches, échelles) --- */
#define SIGIL_ISA 16        /* ligne ouverte (delay line, sans bouclage) */
#define SIGIL_FEHU 17       /* ligne + brindilles qui rejoignent (i->i+2) */
#define SIGIL_ALGIZ 18      /* ligne + feedbacks locaux (branches de vie) */
#define SIGIL_HAGALAZ 19    /* échelle : 2 lignes + barreaux (H répété) */
#define SIGIL_OTHALA 20     /* chaîne de losanges (rhombus tiling) */
#define SIGIL_BINDRUNE 21   /* rune liée : ligne + 8 cordes scellées (seed) */
/* --- Géométrie sacrée (structures validées par recherche web) --- */
#define SIGIL_FLOWER 22     /* Fleur de Vie : treillis hexagonal torique */
#define SIGIL_METATRON 23   /* Cube de Métatron : K13 complet (n=13) */
#define SIGIL_ICHING 24     /* Yi Jing : 64 hexagrammes = hypercube Q6 (n=64) */
#define SIGIL_BAGUA 25      /* Bagua : 8 trigrammes = cube Q3 (n=8) */
#define SIGIL_YETZIRAH 26   /* Sefer Yetzirah : 231 portes = K22 complet (n=22) */
#define SIGIL_LABYRINTH 27  /* labyrinthe unicursal : chemin hamiltonien serpent */
#define SIGIL_KOLAM 28      /* pulli kolam : tore à 8-connectivité (boucles) */
/* --- Sceaux rituels (constantes structurelles documentées) --- */
#define SIGIL_GOETIC 29     /* goétie : anneau + 8 cordes internes (seed = démon) */
#define SIGIL_ENOCHIAN 30   /* Sigillum Dei : anneau + heptagramme + moyeu */
#define SIGIL_VEVE 31       /* vèvè : carrefour (croix) + anneau + paires miroir */
#define SIGIL_OGHAM 32      /* ogham : bâton + rameaux culs-de-sac (dissipation) */
#define SIGIL_ADINKRA 33    /* nkyinkyim : ligne zigzag + torsades longues */
#define SIGIL_NPATTERNS 34

static const char *SIGIL_NAMES[SIGIL_NPATTERNS] = {
    "seal", "pentagram", "ring", "wheel", "grid", "ziggurat", "random",
    "hexagram", "pentacle", "tree", "ouroboros", "triquetra", "ankh",
    "sriyanta", "triskel", "vesica",
    "isa", "fehu", "algiz", "hagalaz", "othala", "bindrune",
    "flower", "metatron", "iching", "bagua", "yetzirah", "labyrinth", "kolam",
    "goetic", "enochian", "veve", "ogham", "adinkra"
};

/** Taille canonique (0 = redimensionnable). */
int sigil_native_n(int pattern) {
    switch (pattern) {
        case SIGIL_TREE: return 10;      /* 10 sephiroth */
        case SIGIL_METATRON: return 13;  /* 13 cercles du Fruit de Vie */
        case SIGIL_ICHING: return 64;    /* 64 hexagrammes */
        case SIGIL_BAGUA: return 8;      /* 8 trigrammes */
        case SIGIL_YETZIRAH: return 22;  /* 22 lettres */
        default: return 0;
    }
}

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
    } else if (pattern == SIGIL_HEXAGRAM) {
        int k = n / 3;                       /* deux triangles entrelacés */
        if (k < 2) k = 2;
        for (int i = 0; i < n; ++i) {
            push_edge(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + k) % n);
        }
    } else if (pattern == SIGIL_PENTACLE) {
        int k = n / 2 - 1;                   /* cercle + étoile */
        if (k < 2) k = 2;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + k) % n);
        }
    } else if (pattern == SIGIL_TREE) {
        /* Arbre de Vie (Kircher) : Kether 0, Chokhmah 1, Binah 2, Chesed 3,
         * Geburah 4, Tiphereth 5, Netzach 6, Hod 7, Yesod 8, Malkuth 9.
         * 22 sentiers (lettres hébraïques) — taille canonique. */
        static const int8_t P[22][2] = {
            {0,1},{0,2},{0,5},{1,2},{1,5},{1,3},{2,5},{2,4},
            {3,4},{3,5},{3,6},{4,5},{4,7},{5,6},{5,8},{5,7},
            {6,7},{6,8},{6,9},{7,8},{7,9},{8,9}
        };
        if (n != 10) return -1;
        for (int k = 0; k < 22; ++k)
            push2(src, dst, max_edges, &m, P[k][0], P[k][1]);
    } else if (pattern == SIGIL_OUROBOROS) {
        for (int i = 0; i < n; ++i)          /* le corps */
            push_edge(src, dst, max_edges, &m, i, (i + 1) % n);
        push_edge(src, dst, max_edges, &m, n - 1, n / 3);  /* la morsure */
    } else if (pattern == SIGIL_TRIQUETRA) {
        int s0 = n / 3, s1 = n / 3, s2 = n - s0 - s1;
        if (s0 < 2 || s1 < 2 || s2 < 2) return -1;
        int b[4]; b[0] = 0; b[1] = s0; b[2] = s0 + s1; b[3] = n;
        for (int r = 0; r < 3; ++r)
            for (int i = b[r]; i < b[r + 1]; ++i)
                push2(src, dst, max_edges, &m, i, b[r] + ((i - b[r] + 1) % (b[r + 1] - b[r])));
        for (int r = 0; r < 3; ++r)          /* les trois boucles se tiennent */
            push2(src, dst, max_edges, &m, b[r + 1] - 1, b[(r + 1) % 3]);
    } else if (pattern == SIGIL_ANKH) {
        int r = (2 * n) / 3;                 /* l'anneau de vie */
        if (r < 4 || n - r < 1) return -1;
        for (int i = 0; i < r; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % r);
        push2(src, dst, max_edges, &m, 0, r / 2);   /* la barre */
        push_edge(src, dst, max_edges, &m, r / 2, r);/* la queue */
        for (int i = r; i < n - 1; ++i)
            push_edge(src, dst, max_edges, &m, i, i + 1);
    } else if (pattern == SIGIL_SRIYANTA) {
        int s0 = n / 3, s1 = n / 3, s2 = n - s0 - s1;
        if (s0 < 4 || s1 < 4 || s2 < 4) return -1;
        int b[4]; b[0] = 0; b[1] = s0; b[2] = s0 + s1; b[3] = n;
        int sz[3] = {s0, s1, s2};
        for (int L = 0; L < 3; ++L) {        /* triangles entrelacés */
            int st = sz[L] / 3;
            if (st < 1) st = 1;
            for (int i = b[L]; i < b[L + 1]; ++i)
                push2(src, dst, max_edges, &m, i, b[L] + ((i - b[L] + st) % sz[L]));
        }
        for (int i = 0; i < s0; ++i) {       /* rayons inter-couches */
            push2(src, dst, max_edges, &m, b[0] + i, b[1] + (i % s1));
            push2(src, dst, max_edges, &m, b[1] + (i % s1), b[2] + (i % s2));
        }
    } else if (pattern == SIGIL_TRISKEL) {
        int per = (n - 1) / 3;               /* 3 bras + moyeu 0 */
        if (per < 1) return -1;
        int base = 1;
        for (int a = 0; a < 3; ++a) {
            int len = (a < 2) ? per : (n - base);  /* le reste au 3e bras */
            if (len < 1) break;
            push_edge(src, dst, max_edges, &m, 0, base);
            for (int i = 0; i < len - 1; ++i)
                push_edge(src, dst, max_edges, &m, base + i, base + i + 1);
            push_edge(src, dst, max_edges, &m, base + len - 1, 0); /* retour */
            base += len;
        }
    } else if (pattern == SIGIL_VESICA) {
        int na = n / 2;                      /* deux anneaux, 2 nœuds communs */
        if (na < 3 || n - na < 2) return -1;
        for (int i = 0; i < na; ++i)         /* anneau A : 0..na-1 */
            push2(src, dst, max_edges, &m, i, (i + 1) % na);
        /* anneau B : 0, 1, na, na+1, ..., n-1, retour à 0 */
        push2(src, dst, max_edges, &m, 0, 1);
        push2(src, dst, max_edges, &m, 1, na);
        for (int i = na; i < n - 1; ++i)
            push2(src, dst, max_edges, &m, i, i + 1);
        push2(src, dst, max_edges, &m, n - 1, 0);
    } else if (pattern == SIGIL_ISA) {
        for (int i = 0; i < n - 1; ++i)      /* bâton ouvert, sans bouclage */
            push_edge(src, dst, max_edges, &m, i, i + 1);
    } else if (pattern == SIGIL_FEHU) {
        for (int i = 0; i < n - 1; ++i) {
            push_edge(src, dst, max_edges, &m, i, i + 1);
            if (i % 3 == 0 && i + 2 < n)     /* brindilles qui rejoignent */
                push_edge(src, dst, max_edges, &m, i, i + 2);
        }
    } else if (pattern == SIGIL_ALGIZ) {
        for (int i = 0; i < n - 1; ++i) {
            push_edge(src, dst, max_edges, &m, i, i + 1);
            if (i % 4 == 0 && i + 3 < n)     /* branches de vie en feedback */
                push_edge(src, dst, max_edges, &m, i + 3, i);
        }
    } else if (pattern == SIGIL_HAGALAZ) {
        for (int i = 0; i + 2 < n; i += 2) { /* deux montants */
            push_edge(src, dst, max_edges, &m, i, i + 2);
            push_edge(src, dst, max_edges, &m, i + 1, i + 3);
        }
        for (int i = 0; i + 1 < n; i += 2)   /* barreaux */
            push2(src, dst, max_edges, &m, i, i + 1);
    } else if (pattern == SIGIL_OTHALA) {
        int k = 0;                           /* chaîne de losanges */
        for (; k + 3 < n; k += 3) {
            push2(src, dst, max_edges, &m, k, k + 1);
            push2(src, dst, max_edges, &m, k, k + 2);
            push2(src, dst, max_edges, &m, k + 1, k + 3);
            push2(src, dst, max_edges, &m, k + 2, k + 3);
        }
        for (; k + 1 < n; ++k)               /* queue éventuelle */
            push_edge(src, dst, max_edges, &m, k, k + 1);
    } else if (pattern == SIGIL_BINDRUNE) {
        for (int i = 0; i < n - 1; ++i)      /* bâton porteur */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int c = 0; c < 8; ++c) {        /* 8 runes scellées (seed) */
            int32_t a = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            int32_t b = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            push_edge(src, dst, max_edges, &m, a, b);
        }
    } else if (pattern == SIGIL_FLOWER) {
        int s = 1;                           /* treillis hexagonal torique */
        while ((s + 1) * (s + 1) <= n) ++s;
        if (s < 3) return -1;
        for (int i = 0; i < n; ++i) {
            int x = i % s, y = (i / s) % s;
            int nb[6][2] = {{1,0},{0,1},{1,-1},{-1,0},{0,-1},{-1,1}};
            for (int d = 0; d < 6; ++d) {
                int xx = (x + nb[d][0] + s) % s, yy = (y + nb[d][1] + s) % s;
                int32_t j = (int32_t)(yy * s + xx);
                if (j < n) push2(src, dst, max_edges, &m, i, j);
            }
        }
    } else if (pattern == SIGIL_METATRON) {
        if (n != 13) return -1;              /* K13 : les 13 cercles reliés */
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                push2(src, dst, max_edges, &m, i, j);
    } else if (pattern == SIGIL_ICHING) {
        if (n != 64) return -1;              /* Q6 : un trait changé = arête */
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 6; ++k) {
                int j = i ^ (1 << k);
                if (j > i) push2(src, dst, max_edges, &m, i, j);
            }
    } else if (pattern == SIGIL_BAGUA) {
        if (n != 8) return -1;               /* Q3 : trigrammes adjacents */
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                int j = i ^ (1 << k);
                if (j > i) push2(src, dst, max_edges, &m, i, j);
            }
    } else if (pattern == SIGIL_YETZIRAH) {
        if (n != 22) return -1;              /* 231 portes : K22 complet */
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                push2(src, dst, max_edges, &m, i, j);
    } else if (pattern == SIGIL_LABYRINTH) {
        int s = 1;                           /* serpent hamiltonien s×s */
        while ((s + 1) * (s + 1) <= n) ++s;
        if (s < 2) return -1;
        int N = s * s, prev = -1;
        for (int y = 0; y < s; ++y)
            for (int xx = 0; xx < s; ++xx) {
                int x = (y % 2 == 0) ? xx : (s - 1 - xx);
                int cur = y * s + x;
                if (prev >= 0) push_edge(src, dst, max_edges, &m, prev, cur);
                prev = cur;
            }
        for (int j = N; j < n; ++j)          /* queue éventuelle */
            push_edge(src, dst, max_edges, &m, j - 1, j);
    } else if (pattern == SIGIL_KOLAM) {
        int s = 1;                           /* tore à 8-connectivité */
        while ((s + 1) * (s + 1) <= n) ++s;
        if (s < 2) s = 2;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + s) % n);
            push2(src, dst, max_edges, &m, i, (i + s + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + s - 1 + n) % n);
        }
    } else if (pattern == SIGIL_GOETIC) {
        for (int i = 0; i < n; ++i)          /* cercle du sceau */
            push_edge(src, dst, max_edges, &m, i, (i + 1) % n);
        for (int c = 0; c < 8; ++c) {        /* glyphe interne (seed=démon) */
            int32_t a = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            int32_t b = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            push2(src, dst, max_edges, &m, a, b);
        }
    } else if (pattern == SIGIL_ENOCHIAN) {
        int k = n / 7;                       /* heptagramme {7/k} */
        if (k < 2) k = 2;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + k) % n);
            if (i > 0) push2(src, dst, max_edges, &m, 0, i);  /* moyeu */
        }
    } else if (pattern == SIGIL_VEVE) {
        /* Carrefour de Legba d'abord, puis anneau, puis paires en miroir
         * (symétrie bilatérale « comme en haut, comme en bas »). */
        push2(src, dst, max_edges, &m, 0, n / 2);
        push2(src, dst, max_edges, &m, n / 4, (3 * n) / 4);
        for (int i = 0; i < n; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
        int k = n / 6;
        if (k < 2) k = 2;
        for (int i = 1; i <= n / 4; ++i) {
            push2(src, dst, max_edges, &m, i, (i + k) % n);
            push2(src, dst, max_edges, &m, (n - i) % n, (n - i - k + n) % n);
        }
    } else if (pattern == SIGIL_OGHAM) {
        /* Bâton (druim) + rameaux culs-de-sac : les impasses dissipent,
         * seules les lignes du bâton propagent. */
        for (int i = 0; i + 1 < n; ++i) {
            if (i % 2 == 0) {
                push_edge(src, dst, max_edges, &m, i, i + 1);  /* rameau */
                if (i + 2 < n)
                    push_edge(src, dst, max_edges, &m, i, i + 2);  /* bâton */
            }
        }
    } else if (pattern == SIGIL_ADINKRA) {
        for (int i = 0; i + 1 < n; ++i)      /* zigzag porteur */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int i = 0; i + 5 < n; ++i)      /* torsades longues (nkyinkyim) */
            push_edge(src, dst, max_edges, &m, i, i + 5);
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
            int nn = sigil_native_n(p) ? sigil_native_n(p) : 64;
            m = sigil_edges(p, nn, 1234, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL pattern %d (%s)\n", p, SIGIL_NAMES[p]); fails++; continue; }
            for (int k = 0; k < m; ++k)
                if (src[k] < 0 || src[k] >= nn || dst[k] < 0 || dst[k] >= nn ||
                    src[k] == dst[k]) {
                    printf("FAIL pattern %d edge %d\n", p, k); fails++; break;
                }
        }
        /* Arbre de Vie : exactement 22 sentiers x2 = 44 arêtes */
        m = sigil_edges(SIGIL_TREE, 10, 1, src, dst, 8 * 128);
        if (m != 44) { printf("FAIL tree m=%d (attendu 44)\n", m); fails++; }
        /* Graphes canoniques : comptes exacts */
        m = sigil_edges(SIGIL_METATRON, 13, 1, src, dst, 8 * 128);
        if (m != 156) { printf("FAIL metatron m=%d (attendu 156)\n", m); fails++; }
        m = sigil_edges(SIGIL_YETZIRAH, 22, 1, src, dst, 8 * 128);
        if (m != 462) { printf("FAIL yetzirah m=%d (attendu 462)\n", m); fails++; }
        m = sigil_edges(SIGIL_ICHING, 64, 1, src, dst, 8 * 128);
        if (m != 384) { printf("FAIL iching m=%d (attendu 384)\n", m); fails++; }
        m = sigil_edges(SIGIL_BAGUA, 8, 1, src, dst, 8 * 128);
        if (m != 24) { printf("FAIL bagua m=%d (attendu 24)\n", m); fails++; }
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
