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
 *  34 FUTHARK   : 24 aînées + 16 jeunes, enchaînées (variant = seed%40)
 *  35 GALDR     : 4 staves islandais (variant = seed%4)
 *  36 VALKNUT   : 3 triangles + triangle central
 *  37 MJOLNIR   : manche + tête triangulaire
 *  38 YGGDRASIL : DAG acyclique (premier motif sans cycle)
 *  39 SHIELDKNOT: 4 anneaux carrés liés
 *  40 BRIGID    : 4 bras + carré tissé central
 *  41 CADUCEUS  : bâton + 2 serpents + barre
 *  42 WEDJAT    : sourcil + boucle + spirale
 *  43 SCARAB    : anneau + 6 pattes + suture
 *  44 DJED      : bâton + 4 barres en tête
 *  45 MONAS     : anneau + croix + croissant
 *  46 ROSECROSS : croix latine + triangle central
 *  47 ALCHEMY   : chaîne de triangles (éléments)
 *  48 TAIJITU   : 2 anneaux + épine S + 2 yeux
 *  49 HAMSA     : noyau 5 + 5 doigts + œil
 *  50 NAZAR     : 3 anneaux + rayons
 *  51 OM        : boucle + arc + croissant
 *  52 CHAOSIGIL : graphe scellé par seed (n=16, intention -> glyphe)
 *  53 FUTHORC   : 24 + 9 anglo-saxonnes, enchaînées (variant = seed%33)
 *  54 ABRAMELIN : grille + accords miroirs (palindromes de lettres)
 *  55 PLANETARY : anneau + moyeu + k cordes (variant = seed%7)
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
/* --- Codex étendu : runes indexées, staves islandais, panthéons --- */
#define SIGIL_FUTHARK 34    /* 24 aînées + 16 jeunes (variant = seed%40) */
#define SIGIL_GALDR 35      /* galdrastafir : 4 staves (variant = seed%4) */
#define SIGIL_VALKNUT 36    /* 3 triangles + triangle central (nœud des tués) */
#define SIGIL_MJOLNIR 37    /* marteau : manche + tête triangulaire */
#define SIGIL_YGGDRASIL 38  /* frêne-monde : DAG acyclique (racines/tronc) */
#define SIGIL_SHIELDKNOT 39 /* nœud bouclier : 4 anneaux carrés liés */
#define SIGIL_BRIGID 40     /* croix de Brigid : 4 bras + carré tissé */
#define SIGIL_CADUCEUS 41   /* caducée : bâton + 2 serpents + barre */
#define SIGIL_WEDJAT 42     /* Œil d'Horus : sourcil + boucle + spirale */
#define SIGIL_SCARAB 43     /* scarabée : anneau + 6 pattes + suture */
#define SIGIL_DJED 44       /* pilier djed : bâton + 4 barres en tête */
#define SIGIL_MONAS 45      /* Monas de Dee : anneau + croix + croissant */
#define SIGIL_ROSECROSS 46  /* Rose-Croix : croix latine + triangle central */
#define SIGIL_ALCHEMY 47    /* éléments : chaîne de triangles */
#define SIGIL_TAIJITU 48    /* yin-yang : 2 anneaux + épine S + 2 yeux */
#define SIGIL_HAMSA 49      /* main : noyau 5 + 5 doigts + œil */
#define SIGIL_NAZAR 50      /* mauvais œil : 3 anneaux + rayons */
#define SIGIL_OM 51         /* Om : boucle + arc + croissant (impression) */
#define SIGIL_CHAOSIGIL 52  /* chaos : graphe scellé par seed (n=16) */
/* --- Derniers manquants du codex --- */
#define SIGIL_FUTHORC 53    /* anglo-saxon : 24 + 9 runes (variant = seed%33) */
#define SIGIL_ABRAMELIN 54  /* carrés Abramelin : grille + accords miroirs */
#define SIGIL_PLANETARY 55  /* Agrippa : anneau + moyeu + k cordes (seed%7) */
#define SIGIL_NPATTERNS 56

static const char *SIGIL_NAMES[SIGIL_NPATTERNS] = {
    "seal", "pentagram", "ring", "wheel", "grid", "ziggurat", "random",
    "hexagram", "pentacle", "tree", "ouroboros", "triquetra", "ankh",
    "sriyanta", "triskel", "vesica",
    "isa", "fehu", "algiz", "hagalaz", "othala", "bindrune",
    "flower", "metatron", "iching", "bagua", "yetzirah", "labyrinth", "kolam",
    "goetic", "enochian", "veve", "ogham", "adinkra",
    "futhark", "galdr", "valknut", "mjolnir", "yggdrasil", "shieldknot",
    "brigid", "caduceus", "wedjat", "scarab", "djed", "monas", "rosecross",
    "alchemy", "taijitu", "hamsa", "nazar", "om", "chaosigil",
    "futhorc", "abramelin", "planetary"
};

/** Taille canonique (0 = redimensionnable). */
int sigil_native_n(int pattern) {
    switch (pattern) {
        case SIGIL_TREE: return 10;      /* 10 sephiroth */
        case SIGIL_METATRON: return 13;  /* 13 cercles du Fruit de Vie */
        case SIGIL_ICHING: return 64;    /* 64 hexagrammes */
        case SIGIL_BAGUA: return 8;      /* 8 trigrammes */
        case SIGIL_YETZIRAH: return 22;  /* 22 lettres */
        case SIGIL_CHAOSIGIL: return 16; /* sigil personnel : petit graphe */
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

/* Indice sûr dans un bloc [b, b+m) : modulo (runes 1D : les côtés n'existent
 * pas, seule la direction haut/bas survit — cf. FUTHARK). */
static int32_t widx(int b, int off, int m) {
    int r = off % m;
    if (r < 0) r += m;
    return (int32_t)(b + r);
}

/* Un glyphe runique dans le bloc [b, b+m) : bâton + traits canoniques.
 * v = 0..23 aînées (Fehu..Othala), 24..39 jeunes (Fe..Yr).
 * Projection 1D honnête : les côtés (gauche/droite) sont perdus, seules
 * les directions haut/bas et les boucles survivent. Les codes quasi
 * identiques mesurent une parenté morphologique réelle (ex. bol partagé
 * raidho/berkano — le glyphe a survécu tel quel à travers les siècles).
 * Convention : brindille HAUTE = arête vers l'avant (x->x+2),
 * brindille BASSE = arête vers l'arrière ((x+2)->x). */
static void rune_glyph(int v, int b, int m,
                       int32_t *src, int32_t *dst, int max_edges, int *mm) {
    int q1 = m / 4, q2 = m / 2, q3 = (3 * m) / 4, e = m - 1;
    /* bâton (toutes sauf kenaz, gebo, jera, sowilo, ingwaz, sol) */
    int nostave = (v == 5 || v == 6 || v == 11 || v == 15 || v == 21 || v == 34);
    if (!nostave)
        for (int i = 0; i < m - 1; ++i)
            push_edge(src, dst, max_edges, mm, b + i, b + i + 1);
    switch (v) {
        case 0:  /* FEHU : 2 brindilles hautes */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q1+2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q2+2,m));
            break;
        case 1:  /* URUZ : diagonale de jonction haute */
            push_edge(src, dst, max_edges, mm, b, widx(b,q2,m));
            break;
        case 2: case 26:  /* THURISAZ/THURS : triangle */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1,m));
            break;
        case 3:  /* ANSUZ : 2 brindilles basses */
            push_edge(src, dst, max_edges, mm, widx(b,q1+2,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2+2,m), widx(b,q2,m));
            break;
        case 4:  /* RAIDHO : bol + jambe */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            break;
        case 5:  /* KENAZ : angle < */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q2,m));
            break;
        case 6:  /* GEBO : X */
            push_edge(src, dst, max_edges, mm, b, widx(b,e,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1,m));
            break;
        case 7:  /* WUNJO : bol pointu */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q1+1,m));
            break;
        case 8:  /* HAGALAZ : parallèle + barre */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push2(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            break;
        case 9: case 31:  /* NAUTHIZ/NAUDR : X médian */
            push_edge(src, dst, max_edges, mm, widx(b,q2-1,m), widx(b,q2+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2+1,m), widx(b,q2-1,m));
            break;
        case 10: case 32: /* ISA/ISS : bâton nu */
            break;
        case 11: /* JERA : deux chevrons convergents */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q2,m));
            break;
        case 12: /* EIHWAZ : zigzag */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1+1,m));
            break;
        case 13: case 28: /* PERTHRO/REID : bol rond */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q1,m));
            break;
        case 14: case 39: /* ALGIZ/YR : paire haute (yr dérive d'algiz) */
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q2+2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2+1,m), widx(b,q2+3,m));
            break;
        case 15: /* SOWILO : éclair */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q2,m));
            break;
        case 16: case 35: /* TIWAZ/TYR : pointe de flèche */
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+2,m), widx(b,q1,m));
            break;
        case 17: case 36: /* BERKANO/BJARKAN : double bol */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q2,m));
            break;
        case 18: /* EHWAZ : diagonale longue */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            break;
        case 19: case 37: /* MANNAZ/MADR : X sommital */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q2,m));
            break;
        case 20: case 38: /* LAGUZ/LOGR : une brindille basse */
            push_edge(src, dst, max_edges, mm, widx(b,q1+2,m), widx(b,q1,m));
            break;
        case 21: /* INGWAZ : losange (cycle de 4) */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1,m));
            break;
        case 22: /* DAGAZ : nœud papillon (2-cycle long) */
            push_edge(src, dst, max_edges, mm, b, widx(b,e,m));
            push_edge(src, dst, max_edges, mm, widx(b,e,m), b);
            break;
        case 23: /* OTHALA : losange + jambes */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q3+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1+2,m));
            break;
        case 24: /* FE jeune : 1 brindille */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q1+2,m));
            break;
        case 25: /* UR jeune : crochet */
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1,m));
            break;
        case 27: /* OSS jeune : brindille basse */
            push_edge(src, dst, max_edges, mm, widx(b,q1+2,m), widx(b,q1,m));
            break;
        case 29: /* KAUN jeune : brindille haute courte */
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q2+2,m));
            break;
        case 30: /* HAGALL jeune : barre simple */
            push2(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            break;
        case 33: /* AR jeune : diagonale */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            break;
        case 34: /* SOL jeune : slash */
            push_edge(src, dst, max_edges, mm, b, widx(b,e,m));
            break;
        /* --- Futhorc anglo-saxon : 9 runes propres (Ac..Gar) --- */
        case 40: /* AC (chêne) : bâton + X latéral */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q2,m));
            break;
        case 41: /* ÆSC (frêne) : bol + brindille haute */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q2+2,m));
            break;
        case 42: /* YR (arc) : bâton + arc qui revient */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q2,m));
            break;
        case 43: /* IOR (anguille) : bâton + double zigzag */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2+1,m), widx(b,q3,m));
            break;
        case 44: /* EAR (terre) : bâton + barre haute */
            push2(src, dst, max_edges, mm, widx(b,1,m), widx(b,3,m));
            break;
        case 45: /* CWEORTH (feu) : bâton + diagonale longue (= ehwaz, récurrent) */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q3,m));
            break;
        case 46: /* CALC (coupe) : bâton + coupe convergente */
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,e,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3+1,m), widx(b,e,m));
            break;
        case 47: /* STAN (pierre) : bâton + losange (= othala + bâton) */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3,m));
            push_edge(src, dst, max_edges, mm, widx(b,q3,m), widx(b,q1+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q1,m));
            break;
        case 48: /* GAR (lance) : bâton + double X */
            push_edge(src, dst, max_edges, mm, widx(b,q1,m), widx(b,q2+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q1+1,m), widx(b,q2,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2,m), widx(b,q3+1,m));
            push_edge(src, dst, max_edges, mm, widx(b,q2+1,m), widx(b,q3,m));
            break;
        default: break;
    }
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
    } else if (pattern == SIGIL_FUTHARK) {
        /* Inscription : k runes enchaînées (1..3 selon seed), rune j =
         * (v0 + 7j) % 40 — v0 = seed%40 : 0..23 aînées, 24..39 jeunes. */
        int v0 = (int)(seed % 40);
        int k = 1 + (int)(seed % 3);
        int base = 0;
        for (int j = 0; j < k; ++j) {
            int sz = (j < k - 1) ? (n / k) : (n - base);
            if (sz >= 4)
                rune_glyph((v0 + 7 * j) % 40, base, sz, src, dst, max_edges, &m);
            else
                for (int i = base; i < base + sz - 1; ++i)
                    push_edge(src, dst, max_edges, &m, i, i + 1);
            if (j < k - 1 && base + sz < n)  /* chaîne inter-runes */
                push_edge(src, dst, max_edges, &m, base + sz - 1, base + sz);
            base += sz;
        }
    } else if (pattern == SIGIL_GALDR) {
        if (n < 16) return -1;
        int g = (int)(seed % 4), L = n / 8;
        if (g <= 1) {
            /* ÆGISHJALMUR (0) : moyeu + 8 bras à barbelures ;
             * VEGVISIR (1) : 8 bras + anneau des pointes. */
            for (int a = 0; a < 8; ++a) {
                int nb = 1 + a * L, ne = nb + L - 1;
                if (ne >= n) ne = n - 1;
                push_edge(src, dst, max_edges, &m, 0, nb);
                for (int i = nb; i < ne; ++i)
                    push_edge(src, dst, max_edges, &m, i, i + 1);
                if (g == 0 && ne - 2 >= nb)  /* barbelure (T-tip) */
                    push_edge(src, dst, max_edges, &m, ne, ne - 2);
            }
            if (g == 1) {                    /* anneau des pointes */
                int tips[8], nt = 0;
                for (int a = 0; a < 8; ++a) {
                    int ne = 1 + a * L + L - 1;
                    tips[nt++] = (ne < n) ? ne : (n - 1);
                }
                for (int a = 0; a < 8; ++a)
                    push2(src, dst, max_edges, &m, tips[a], tips[(a + 1) % 8]);
            }
        } else {
            /* GAPALDUR (2) : deux bâtons + barreaux ; GINFAXI (3) : + X. */
            for (int i = 0; i + 2 < n; i += 2) {
                push_edge(src, dst, max_edges, &m, i, i + 2);
                push_edge(src, dst, max_edges, &m, i + 1, i + 3);
            }
            for (int i = 0; i + 3 < n; i += 4) {
                if (g == 2) push2(src, dst, max_edges, &m, i, i + 1);
                else {
                    push_edge(src, dst, max_edges, &m, i, i + 3);
                    push_edge(src, dst, max_edges, &m, i + 2, i + 1);
                }
            }
        }
    } else if (pattern == SIGIL_VALKNUT) {
        if (n < 9) return -1;                /* 3 triangles + triangle central */
        push2(src, dst, max_edges, &m, 0, n / 3);
        push2(src, dst, max_edges, &m, n / 3, (2 * n) / 3);
        push2(src, dst, max_edges, &m, (2 * n) / 3, 0);
        int b[4]; b[0] = 3; b[1] = 3 + n / 3; b[2] = 3 + 2 * (n / 3); b[3] = n;
        for (int r = 0; r < 3; ++r) {
            int sz = b[r + 1] - b[r];
            if (sz < 3) continue;
            int st = sz / 3;
            if (st < 1) st = 1;
            for (int i = b[r]; i < b[r + 1]; ++i)
                push2(src, dst, max_edges, &m, i, b[r] + ((i - b[r] + st) % sz));
        }
    } else if (pattern == SIGIL_MJOLNIR) {
        int h = (3 * n) / 4;                 /* manche */
        if (h < 2 || n - h < 3) return -1;
        for (int i = 0; i < h - 1; ++i)
            push_edge(src, dst, max_edges, &m, i, i + 1);
        push_edge(src, dst, max_edges, &m, h - 1, h);  /* tête triangulaire */
        push2(src, dst, max_edges, &m, h, h + 1);
        push2(src, dst, max_edges, &m, h + 1, h + 2);
        push2(src, dst, max_edges, &m, h + 2, h);
        for (int i = h + 3; i < n; ++i)      /* garde éventuelle */
            push_edge(src, dst, max_edges, &m, i - 1, i);
    } else if (pattern == SIGIL_YGGDRASIL) {
        /* DAG acyclique : 3 racines convergent, tronc, 3 cimes divergent.
         * Premier motif sans aucun cycle : teste la mémoire purement
         * feedforward (attendue ~0). */
        int t0 = n / 3, t1 = (2 * n) / 3;
        if (t0 < 2 || t1 - t0 < 2 || n - t1 < 2) return -1;
        for (int a = 0; a < 3; ++a) {        /* racines -> base */
            int rb = a * (t0 / 3), re = (a < 2) ? ((a + 1) * (t0 / 3)) : t0;
            for (int i = rb; i < re - 1; ++i)
                push_edge(src, dst, max_edges, &m, i, i + 1);
            if (re - 1 >= rb) push_edge(src, dst, max_edges, &m, re - 1, t0);
        }
        for (int i = t0; i < t1 - 1; ++i)    /* tronc */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int a = 0; a < 3; ++a) {        /* cimes depuis le sommet */
            int cb = t1 + a * ((n - t1) / 3), ce = (a < 2) ? (t1 + (a + 1) * ((n - t1) / 3)) : n;
            if (cb < n) push_edge(src, dst, max_edges, &m, t1 - 1, cb);
            for (int i = cb; i < ce - 1; ++i)
                push_edge(src, dst, max_edges, &m, i, i + 1);
        }
    } else if (pattern == SIGIL_SHIELDKNOT) {
        int q = n / 4;                       /* 4 anneaux carrés liés */
        if (q < 2) return -1;
        for (int r = 0; r < 4; ++r) {
            int b0 = r * q, b1 = (r < 3) ? ((r + 1) * q) : n;
            for (int i = b0; i < b1; ++i)
                push2(src, dst, max_edges, &m, i, b0 + ((i - b0 + 1) % (b1 - b0)));
            push2(src, dst, max_edges, &m, b1 - 1, b1 % n);  /* carré lié */
        }
    } else if (pattern == SIGIL_BRIGID) {
        int L = n / 4;                       /* 4 bras + carré tissé */
        if (L < 2) return -1;
        int mid[4];
        for (int a = 0; a < 4; ++a) {
            int b0 = 1 + a * L, b1 = (a < 3) ? (1 + (a + 1) * L) : n;
            push_edge(src, dst, max_edges, &m, 0, b0);
            for (int i = b0; i < b1 - 1; ++i)
                push_edge(src, dst, max_edges, &m, i, i + 1);
            mid[a] = b0 + (b1 - b0) / 2;
        }
        for (int a = 0; a < 4; ++a)          /* le carré tissé central */
            push2(src, dst, max_edges, &m, mid[a], mid[(a + 1) % 4]);
    } else if (pattern == SIGIL_CADUCEUS) {
        for (int i = 0; i < n - 1; ++i)      /* bâton */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int i = 0; i + 2 < n; i += 2) { /* serpents jumeaux */
            push_edge(src, dst, max_edges, &m, i, i + 2);
            push_edge(src, dst, max_edges, &m, i + 1, i + 3);
        }
        for (int i = 0; i + 3 < n; i += 3)   /* enlacements */
            push2(src, dst, max_edges, &m, i, i + 3);
        if (n > 4) push2(src, dst, max_edges, &m, n - 3, n - 1);  /* ailes */
    } else if (pattern == SIGIL_WEDJAT) {
        int t0 = n / 3, t1 = (2 * n) / 3;    /* sourcil / œil / spirale */
        if (t0 < 2 || t1 - t0 < 3 || n - t1 < 2) return -1;
        for (int i = 0; i < t0 - 1; ++i)     /* sourcil */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int i = t0; i < t1; ++i)        /* boucle de l'œil */
            push2(src, dst, max_edges, &m, i, t0 + ((i - t0 + 1) % (t1 - t0)));
        push_edge(src, dst, max_edges, &m, t0 - 1, t0);
        for (int i = t1; i < n - 1; ++i)     /* spirale de la joue */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int i = t1; i + 3 < n; i += 4)
            push_edge(src, dst, max_edges, &m, i + 3, i);
    } else if (pattern == SIGIL_SCARAB) {
        int r = (2 * n) / 3;                 /* corps en anneau */
        if (r < 4 || n - r < 7) return -1;
        for (int i = 0; i < r; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % r);
        push2(src, dst, max_edges, &m, 0, r / 2);   /* suture des élytres */
        for (int l = 0; l < 6; ++l) {        /* 6 pattes (culs-de-sac) */
            int hip = (l * r) / 6, leg = r + l;
            if (leg < n) push_edge(src, dst, max_edges, &m, hip, leg);
        }
        if (r + 6 < n)                       /* tête */
            push_edge(src, dst, max_edges, &m, r / 4, r + 6);
    } else if (pattern == SIGIL_DJED) {
        for (int i = 0; i < n - 1; ++i)      /* pilier */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int h = 0; h < 4; ++h) {        /* 4 barres en tête */
            int at = n - 2 - h * (n / 8);
            if (at > n / 2 && at + 2 < n)
                push_edge(src, dst, max_edges, &m, at, at + 2);
        }
    } else if (pattern == SIGIL_MONAS) {
        int t0 = n / 3, t1 = (2 * n) / 3;    /* soleil + croix + croissant */
        if (t0 < 2 || t1 - t0 < 3 || n - t1 < 2) return -1;
        for (int i = t0; i < t1; ++i)        /* anneau solaire */
            push2(src, dst, max_edges, &m, i, t0 + ((i - t0 + 1) % (t1 - t0)));
        push2(src, dst, max_edges, &m, t0, t1 - 1);      /* croix */
        push2(src, dst, max_edges, &m, (t0 + t1) / 2, 0);
        for (int i = 0; i < t0 - 1; ++i)     /* croissant ouvert */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        for (int i = t1; i < n - 1; ++i)     /* base */
            push_edge(src, dst, max_edges, &m, i, i + 1);
        push_edge(src, dst, max_edges, &m, t0 - 1, t0);
        push_edge(src, dst, max_edges, &m, t1 - 1, t1);
    } else if (pattern == SIGIL_ROSECROSS) {
        int c = n / 3;                       /* croix latine + rose */
        for (int i = 0; i < n - 1; ++i)
            push_edge(src, dst, max_edges, &m, i, i + 1);
        if (c > 2 && c + 2 < n) {
            push2(src, dst, max_edges, &m, c - 2, c + 2);  /* barre */
            push_edge(src, dst, max_edges, &m, c - 1, c);  /* rose */
            push_edge(src, dst, max_edges, &m, c, c + 1);
            push_edge(src, dst, max_edges, &m, c + 1, c - 1);
        }
    } else if (pattern == SIGIL_ALCHEMY) {
        int k = 0;                           /* chaîne de triangles */
        for (; k + 2 < n; k += 2) {
            push_edge(src, dst, max_edges, &m, k, k + 1);
            push_edge(src, dst, max_edges, &m, k + 1, k + 2);
            push_edge(src, dst, max_edges, &m, k + 2, k);
        }
        for (; k + 1 < n; ++k)
            push_edge(src, dst, max_edges, &m, k, k + 1);
    } else if (pattern == SIGIL_TAIJITU) {
        int h = n / 2;                       /* 2 anneaux + épine S + yeux */
        if (h < 3 || n - h < 3) return -1;
        for (int i = 0; i < h; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % h);
        for (int i = h; i < n; ++i)
            push2(src, dst, max_edges, &m, i, h + ((i - h + 1) % (n - h)));
        push_edge(src, dst, max_edges, &m, h - 1, h);      /* épine S */
        push_edge(src, dst, max_edges, &m, n - 1, 0);
        push_edge(src, dst, max_edges, &m, h / 2, h / 2 + 2);      /* yeux */
        push_edge(src, dst, max_edges, &m, h + (n - h) / 2, h + (n - h) / 2 + 2);
    } else if (pattern == SIGIL_HAMSA) {
        for (int i = 0; i < 5; ++i)          /* paume : anneau de 5 */
            push2(src, dst, max_edges, &m, i % 5, (i + 1) % 5);
        push2(src, dst, max_edges, &m, 0, 2);              /* œil */
        int per = (n - 5) / 5;               /* 5 doigts */
        for (int f = 0; f < 5; ++f) {
            int b0 = 5 + f * per, b1 = (f < 4) ? (5 + (f + 1) * per) : n;
            if (b0 < n) push_edge(src, dst, max_edges, &m, f, b0);
            for (int i = b0; i < b1 - 1; ++i)
                push_edge(src, dst, max_edges, &m, i, i + 1);
        }
    } else if (pattern == SIGIL_NAZAR) {
        int s0 = n / 3, s1 = n / 3, s2 = n - s0 - s1;  /* 3 anneaux */
        if (s0 < 2 || s1 < 2 || s2 < 2) return -1;
        int b[4]; b[0] = 0; b[1] = s0; b[2] = s0 + s1; b[3] = n;
        for (int r = 0; r < 3; ++r)
            for (int i = b[r]; i < b[r + 1]; ++i)
                push2(src, dst, max_edges, &m, i, b[r] + ((i - b[r] + 1) % (b[r + 1] - b[r])));
        for (int i = 0; i < s0; ++i)         /* rayons */
            push2(src, dst, max_edges, &m, b[0] + i, b[1] + (i % s1));
    } else if (pattern == SIGIL_OM) {
        int h = n / 2;                       /* boucle + arc + croissant */
        if (h < 3 || n - h < 2) return -1;
        for (int i = 0; i < h; ++i)
            push2(src, dst, max_edges, &m, i, (i + 1) % h);
        for (int i = h; i < n - 1; ++i)
            push_edge(src, dst, max_edges, &m, i, i + 1);
        push_edge(src, dst, max_edges, &m, h - 1, h);
        if (n - h > 3)
            push_edge(src, dst, max_edges, &m, n - 1, h + 1);  /* croissant */
    } else if (pattern == SIGIL_CHAOSIGIL) {
        if (n != 16) return -1;              /* sigil personnel scellé */
        int target = (5 * n) / 2, guard = 0;
        while (m < target && guard++ < target * 20) {
            int32_t a = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            int32_t b = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            push_edge(src, dst, max_edges, &m, a, b);
        }
    } else if (pattern == SIGIL_FUTHORC) {
        /* Futhorc : 24 aînées (mêmes glyphes, conservés) + 9 propres
         * (Ac, Æsc, Yr, Ior, Ear, Cweorth, Calc, Stan, Gar : v = 40..48).
         * Inscription enchaînée comme FUTHARK. */
        int v0 = (int)(seed % 33);
        int k = 1 + (int)(seed % 3);
        int base = 0;
        for (int j = 0; j < k; ++j) {
            int w = (v0 + 7 * j) % 33;
            int vv = (w < 24) ? w : (40 + (w - 24));
            int sz = (j < k - 1) ? (n / k) : (n - base);
            if (sz >= 4)
                rune_glyph(vv, base, sz, src, dst, max_edges, &m);
            else
                for (int i = base; i < base + sz - 1; ++i)
                    push_edge(src, dst, max_edges, &m, i, i + 1);
            if (j < k - 1 && base + sz < n)
                push_edge(src, dst, max_edges, &m, base + sz - 1, base + sz);
            base += sz;
        }
    } else if (pattern == SIGIL_ABRAMELIN) {
        /* Carrés de lettres palindromes : grille torique + accords miroirs
         * (chaque nœud lié à son miroir sur les deux axes = la contrainte
         * palindrome du carré magique de lettres). */
        int s = 1;
        while ((s + 1) * (s + 1) <= n) ++s;
        if (s < 2) s = 2;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            push2(src, dst, max_edges, &m, i, (i + s) % n);
            int x = i % s, y = (i / s) % s;
            int32_t mx = (int32_t)(y * s + (s - 1 - x));
            int32_t my = (int32_t)(((s - 1 - y) * s + x));
            if (mx < n) push2(src, dst, max_edges, &m, i, mx);
            if (my < n && my != mx) push2(src, dst, max_edges, &m, i, my);
        }
    } else if (pattern == SIGIL_PLANETARY) {
        /* Sceaux planétaires d'Agrippa (7) : anneau + moyeu (les sphères
         * tournent autour du centre) + k cordes scellées, k = 3 + planète.
         * variant = seed%7 : 0 Saturne .. 6 Lune. */
        int v = (int)(seed % 7), k = 3 + v;
        for (int i = 0; i < n; ++i) {
            push2(src, dst, max_edges, &m, i, (i + 1) % n);
            if (i > 0) push2(src, dst, max_edges, &m, 0, i);
        }
        for (int c = 0; c < k; ++c) {
            int32_t a = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            int32_t b = (int32_t)(lcg_next(&rng) % (uint32_t)n);
            push2(src, dst, max_edges, &m, a, b);
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
        /* Futhark : les 40 runes (24 + 16) génèrent toutes quelque chose */
        for (int v = 0; v < 40; ++v) {
            m = sigil_edges(SIGIL_FUTHARK, 64, (uint32_t)v, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL futhark v=%d\n", v); fails++; }
        }
        /* Galdrastafir : les 4 staves */
        for (int g = 0; g < 4; ++g) {
            m = sigil_edges(SIGIL_GALDR, 64, (uint32_t)g, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL galdr g=%d\n", g); fails++; }
        }
        /* Futhorc : les 33 runes (24 + 9) */
        for (int v = 0; v < 33; ++v) {
            m = sigil_edges(SIGIL_FUTHORC, 64, (uint32_t)v, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL futhorc v=%d\n", v); fails++; }
        }
        /* Planétaires : les 7 sceaux */
        for (int v = 0; v < 7; ++v) {
            m = sigil_edges(SIGIL_PLANETARY, 64, (uint32_t)v, src, dst, 8 * 128);
            if (m <= 0) { printf("FAIL planetary v=%d\n", v); fails++; }
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
