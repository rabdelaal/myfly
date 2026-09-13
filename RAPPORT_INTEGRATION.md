# Rapport d'intégration — Papier Cell 2026 (dimorphisme MaleCNS) & Recurrent Looped Transformer

**Date** : 13 septembre 2026
**Objet** : comment intégrer et implémenter dans FlyChess (1) les découvertes et données du papier
*« Sexual dimorphism in the complete Drosophila male central nervous system connectome »*
(Cell 189:5504–5526, 3 sept. 2026, Janelia/MRC-LMB) et (2) les principes d'architecture du
**Recurrent Looped Transformer (RLT)** (Zhang, rapport technique sept. 2026).

---

## 1. État des lieux du projet

FlyChess simule le connectome MaleCNS v1.0 en LIF (le vrai, 183 490 neurones / ~26 M synapses)
et l'utilise comme cerveau d'un Tamagotchi qui joue aux échecs. Ce qui existe déjà :

| Composant | Fichier | État |
|---|---|---|
| Conversion feather → npz | `scripts/prepare_data.py` | n'exploite que `superclass`, `status`, `somaLocation`, NT |
| Chargement connectome | `backend/data_loader.py` | W signé CSR + masques sensoriel/moteur/descendant + GRN/MN9 |
| Simulation LIF batchée | `backend/brain.py` | modes `current` et `alpha` (Shiu), kernel C natif vérifié |
| Encodage échiquier | `backend/encoding.py` | projection **aléatoire** 788 features → sensoriels |
| Readout | `backend/readout.py` + `train_readout.py` | **linéaire**, règle à 3 facteurs ; **ne généralise pas** (ρ ≈ 0,14, réponse quasi rank-1) |
| Théâtre live | `backend/live.py`, `frontend/connectome.js` | populations = superclasses uniquement |

**Le problème central du projet** (documenté honnêtement dans le README) : sous notre régime
LIF, la réponse du connectome est quasi **rank-1** — un niveau d'activité global, sans
information position-quality linéairement décodable. Les deux sources de ce rapport attaquent
exactement ce problème, par des angles complémentaires :

- **Le papier Cell** dit *où* et *comment* l'information voyage : voies sensorielles→moteurs
  annotées bout en bout, neurones descendants = intégrateurs, dimorphisme concentré dans les
  couches profondes, hotspots fru/dsx.
- **RLT** dit *comment lire et entraîner* au-dessus d'un état récurrent : un décodeur avec
  profondeur temporelle (SWA + feedback du dernier état caché) et une exécution identique
  train/inference.

---

## 2. Partie A — Intégrer le papier Cell 2026

### A0. Découverte clé : nous avons déjà les données (sans les utiliser)

Le dump `data/body-annotations.feather` (déjà téléchargé, MaleCNS v1.0) contient **déjà les
annotations produites pour ce papier**. Vérifié sur notre copie (211 577 lignes) :

| Colonne | Valeurs présentes chez nous | Rappel papier (Table 1) |
|---|---|---|
| `dimorphism` | `male-specific` 1 258, `sexually dimorphic` 771, `potentially male-specific` 162, `potentially sexually dimorphic` 177 | 1 420 male-specific (0,8 %), 948 dimorphic (0,5 %) — **accord exact** une fois les « potentially » additionnés |
| `fruDsx` | `fru_high` 2 611, `fru_low` 1 989, `coexpress_high` 193, `dsx_high` 138, … | 4 858 fru+ (2 804 haut conf.), 412 dsx+, 258 co-exprimés |
| `type` / `subclass` / `class` | 11 710 types ; `class` = visual, Kenyon_Cell, CX, olfactory, gustatory, ALPN, DAN, MBON… | l'atlas de référence « Rosetta stone » |
| `receptorType` | sous-types de récepteurs sensoriels | base des canaux modaux (A3) |
| `entryNerve` / `exitNerve` | nerf d'entrée/sortie | câblage périphérique sensoriel→muscle |
| `somaNeuromere`, `somaSide`, hémilignées (`trumanHl`, `itoleeHl`, `birthtime`) | origine développementale | 8 hémilignées produisent >50 % des neurones sex-spécifiques |

**Conclusion : la Partie A coûte surtout du code, pas de téléchargement.**

### A1. Étendre `prepare_data.py` et `data_loader.py` (½ jour)

Ajouter au npz (arrays alignés sur `neuron_ids`, via le même lookup `searchsorted` que
`build_nt_signs`) :

```python
# scripts/prepare_data.py — dans build_annotations(), extraire aussi :
#   types        = ann["type"]            (11 710 types cellules)
#   dimorphism   = ann["dimorphism"]      (4 valeurs + None)
#   frudsx       = ann["fruDsx"]          (6 valeurs + None)
#   receptor     = ann["receptorType"]
#   entry_nerve  = ann["entryNerve"] / exit_nerve = ann["exitNerve"]
#   subclass     = ann["subclass"]
# et dans main() → np.savez_compressed(..., types=..., dimorphism=..., frudsx=...)
```

Côté `data_loader.py`, exposer des masques prêts à l'emploi (ils alimentent A2–A5) :

```python
conn["is_male_specific"] = dimorphism == "male-specific"
conn["is_dimorphic"]     = dimorphism == "sexually dimorphic"
conn["is_fru"]           = np.isin(frudsx, ["fru_high", "fru_low", "coexpress_high", "coexpress_low"])
conn["is_dsx"]           = np.isin(frudsx, ["dsx_high", "dsx_low", "coexpress_high", "coexpress_low"])
conn["is_hotspot"]       = conn["is_male_specific"] | conn["is_dimorphic"] | conn["is_fru"] | conn["is_dsx"]
```

⚠️ `type`/`fruDsx` sont des chaînes : encoder en `int16` + table de vocab JSON séparée
(`data/label_maps.json`) pour garder le npz compact.

### A2. Encodage sensoriel guidé par la biologie (le changement le plus prometteur) — 1 à 2 jours

Le papier montre que **le périphérique sensoriel est isomorphe et modulairement organisé**
(53 ORN + 8 TRN types, 6 sous-classes gustatives, rétinotopie des VPN…), et que l'information
est « re-routée » vers les circuits dimorphes par les **PNs de second ordre**. Notre encodage
actuel (`encoding.py`) projette les features du plateau sur *tous* les sensoriels au hasard —
c'est contradictoire avec cette organisation, et vraisemblablement une cause du rank-1.

**Proposition** : remplacer la projection aléatoire par des **canaux modaux** — chaque « sens »
reçoit une facette de la position :

| Canal | Population cible (via `class`/`subclass`/`receptorType`) | Features échecs |
|---|---|---|
| **Visuel** | `class=visual` (6 091 neur.), olfensive exclue ; sensoriels `ol_sensory` | pattern des pièces (les « voir ») |
| **Mécanosensoriel** | `mechanosensory*` (~5 700) | menaces, coups légaux, échec (le « contact ») |
| **Gustatif** | `gustatory` (1 428) | matériel (la « saveur » de la position) |
| **Olfactif** | `class=olfactory` (2 639, ALPN) | position du roi adverse (le « phéromone ») |

Implémentation : `BoardEncoder` construit un vecteur (n_sensory,) **structuré** (bloc par
modalité, gain calibré par canal) au lieu de `feats @ proj` global ; conserver la projection
aléatoire *à l'intérieur* de chaque canal (elle respecte la norme constante déjà calibrée,
`STIM_TARGET_NORM=450`). La norme par canal devient un hyperparamètre par modalité.

**Test de non-régression avant tout** : le réflexe d'alimentation (Shiu) et la courbe
dose-réponse (`exp_feeding_reflex.py`, `feeding_dose_response.npy`) doivent rester intacts —
le canal gustatif utilise d'ailleurs les vraies `tpGRN` déjà identifiées via `type` dans
`data_loader._add_special_neurons`.

### A3. Readout sur les vraies voies descendantes — 1 à 2 jours

Le papier : les **neurones descendants (DN) sont les intégrateurs du goulot cervical**
(analyse max-flow sensor→moteur ; DNp44 = multi-modale, DNa02/aSP22 = poursuite visuelle, etc.),
et les connectivités sensor→DN→pré-moteur sont annotées. Nous lisons déjà
`descending_last` dans le readout, mais :

1. **Restreindre/remplacer le readout par les DN à fort max-flow** identifiés dans Data S1 du
   papier (types DN/AN connus, retrouvables via `type` dans nos annotations) plutôt que les
   1 314 descendants bruts ;
2. Lire aussi l'activité des **hotspots fru/dsx** (A1) — le papier montre que c'est là que
   l'information est re-routée ;
3. Ajouter la **trajectoire temporelle** (voir B1) : `brain.run` ne retourne que
   `motor_mean`/`descending_last` ; ajouter `return_history=True` qui empile
   `[motor, descending, hotspot, sensory]` par pas → tenseur (T, n_read) ≈ 60 × ~4 000, négligeable.

### A4. Le dimorphisme dans le produit (Tamagotchi + théâtre) — 1 à 2 jours

- **Théâtre live** (`live.py` + `frontend/connectome.js`) : deux nouvelles populations dans le
  HUD — « ♂ spécifiques » (1 420) et « fru/dsx » (~4 900) — et une couche de coloration
  optionnelle des 1500 synapses affichées quand l'un des deux partenaires est un hotspot. Le
  papier prédit un effet visuel spectaculaire : les hotspots forment des **sous-réseaux denses
  et interconnectés** (« hotspots » du Fig. 7), ils s'allumeront en grappes.
- **Gameplay** : le papier offre un mécanisme crédible à deux nouveaux comportements —
  (a) *sérendipité sexuelle* : un stimulus « mouche femelle » (canal olfactif DA1/VA1v, cf. §A2)
  active les hotspots → la mouche « chante » (animation aile, +humeur, −énergie) ;
  (b) *agressivité* : stimulus mécanosensoriel répété → route dimorphe agression. Les deux
  circuits (pC1, aSP-*, vPN1) sont nommés dans le papier et présents dans `type`.
- **Pédagogie** : une fiche « dimorphisme » dans l'UI (les chiffres Table 1 + 90,4 % des
  male-specific sont fru+/dsx+).

### A5. Ce que le papier **ne** permet pas d'intégrer (limites, à documenter)

- Les **arêtes dimorphes** (5,8 % des connexions centrales) sont une *analyse* du papier
  (filtrage bruit + t-stat sur deux connectomes), pas un dump téléchargeable ; on ne peut pas
  les reproduire sans le connectome femelle FAFB appairé — hors de portée ici.
- Comparaison mâle/femelle complète : FANC/FANC-57 partiellement corrigée ; BANC pas encore prête.
- Le connectome est un instantané d'un seul individu mâle : les différences « potentiellement
  dimorphes » sont d'ailleurs marquées comme telles dans la colonne `dimorphism` — garder les
  4 catégories distinctes dans l'UI, ne pas fusionner.

---

## 3. Partie B — Intégrer les principes RLT

### B0. Cadrage honnête

RLT (repo `yifanzhang-pro/recurrent-looped-tranformer`) est un **rapport technique sans code ni
benchmark** ; l'auteur lui-même écrit que « les gains de raisonnement, hardware et RL scaling
sont des objectifs de recherche, pas des résultats mesurés ». On n'implémente donc **pas RLT
lui-même**, on transpose ses trois principes à notre problème : décoder un état récurrent
(le cerveau LIF) avec un décodeur à profondeur temporelle, et unifier exécution
d'entraînement et d'inférence.

### B1. Correspondance conceptuelle

| Concept RLT | Équivalent FlyChess | Ce qu'on en fait |
|---|---|---|
| Encodeur causal parallèle → mémoire KV globale | Le connectome LIF lui-même (le « monde » traité en parallèle pour B candidats) | figé (déjà notre moteur) |
| Décoder récurrent : `s_t = D(Merge(e_t, s_{t-1}); mémoire, SWA)` | Lire la trajectoire du réseau pas à pas | **LoopedReadout** (B2) |
| Profondeur temporelle infinie (t × L_D blocs) | 60 pas × profondeur du décodeur | l'info peut être dans la *dynamique*, pas la moyenne — attaque directe du rank-1 |
| SWA (fenêtre glissante) | fenêtre de ~16 pas de simulation | garder la phase (oscillations, bouffées) |
| État complet jamais réinitialisé au prompt boundary | pas de reset entre « position → évaluation » | le stimulus précédent influence la lecture (déjà vrai biologiquement) |
| Une seule exécution train/inférence | même simulateur, même encodage, même décodeur | corrige notre asymétrie actuelle : la règle à 3 facteurs n'entraîne que `fc` sur des *moyennes*, alors que l'inférence décode aussi des moyennes — avec B2 les deux consomment la trajectoire complète |
| Co-design RL : replay exact sous poids courants | la simulation LIF est **déterministe** étant donné les poids | REINFORCE naturel (B4) |
| Co-design hardware : encoder parallèle / décoder récurrent | 20 coups candidats en batch parallèle, décodeur séquentiel léger | déjà notre architecture ; le décodeur ajouté coûte ~0 |

### B2. Phase B2 (cœur) : `LoopedReadout` — 2 à 3 jours

Nouveau fichier `backend/readout_looped.py`. Le décodeur consomme x_t = frame de lecture du
pas t (cf. A3.3) et porte un état récurrent + une fenêtre SWA :

```python
class LoopedCell(nn.Module):
    """Décodeur inspiré RLT : merge(entrée, état précédent) → GRU minimal,
    + attention sur une fenêtre glissante des derniers W états cachés."""
    def __init__(self, d_in, d_h=128, window=16):
        super().__init__()
        self.merge = nn.Linear(d_in + d_h, d_h)          # Merge(e_t, s_{t-1})
        self.gru   = nn.GRUCell(d_h, d_h)                # récurrence (1 bloc "L_D")
        self.swa   = nn.Linear(window * d_h, d_h)        # mémoire locale SWA
        self.head  = nn.Linear(d_h, 1)                   # score scalaire

    def forward(self, x):                                # x: (T, B, d_in)
        B, W = x.shape[1], self.swa.in_features // self.gru.hidden_size
        s = x.new_zeros(B, self.gru.hidden_size)         # s_0 = 0 (pas de reset
        buf = []                                         #  entre positions : s porte
        for t in range(x.shape[0]):                      #  le contexte de la partie)
            h = torch.tanh(self.merge(torch.cat([x[t], s], -1)))
            s = self.gru(h, s)
            buf.append(s)
            w = torch.stack(buf[-W:], 1).flatten(1)      # fenêtre SWA
            pad = self.gru.hidden_size * (W - len(buf))  # padding gauche
            if pad > 0: w = torch.cat([s.new_zeros(B, pad), w], 1)
            yield self.head(torch.tanh(self.swa(w))).squeeze(-1)  # score courant
```

Entraînement (`train_readout_looped.py`, fork de `train_readout.py`) : **full BPTT** par
autograd PyTorch sur la trajectoire (le connectome reste gelé, `@torch.no_grad()` côté
`brain.run` — on n'a pas besoin de traverser la simulation, cf. B3), loss MSE contre le score
Stockfish, *supervision à chaque pas* (le score final est le pas T) — exactement la philosophie
« supervise every valid next-token target » de RLT. Le même objet sert à l'inférence
(`main.py`) : une seule exécution.

**Protocole de validation inchangé** (c'est notre force) : positions held-out, ρ de Spearman,
checkpoint du meilleur. Critère de succès : **ρ > 0,3 sur n ≥ 25 positions** (baseline
linéaire : 0,14 ± 0,2), et supériorité au readout aléatoire confirmée sur 3 seeds.
Critère d'abandon honnête : si ρ reste < 0,2 avec trajectoire + canaux modaux, le goulot est
la simulation LIF (pas le décodeur) → passer à B3 ou revoir le régime synaptique.

### B3. Phase optionnelle : simulation différentiable (surrogate gradients) — 3 à 5 jours

RLT passe des gradients complets à travers tout l'état (`full BPTT`; « detaching any of these
is a gradient approximation »). Chez nous, l'équivalent serait d'entraîner **l'encodage** (la
projection aléatoire → apprise) en traversant la simulation avec des surrogate gradients
(spike = sigmoid(β(V−V_th))). Faisable : 26 M synapses × 60 pas × backward en batch 8 —
cher sur CPU (estimation : plusieurs heures/époque), donc à ne tenter qu'après B2, et en
réduisant `n_steps`. Gain attendu : un encodage qui place l'information dans le pattern (notre
correctif `STIM_TARGET_NORM` l'a rendu *possible*, l'apprentissage le rendrait *optimal*).

### B4. Phase optionnelle : co-design RL (au-delà du score Stockfish) — 2 à 3 jours

Aujourd'hui la mouche ne « choisit » pas : ses 20 candidats sont scorés et on prend le max
(bruit d'humeur ajouté). Transposition du principe RL de RLT :

1. **Politique explicite** : softmax(β · scores des candidats) → échantillonnage d'un coup ;
2. **Récompense de fin de partie** (gain/nulle/perte) au lieu du seul score Stockfish
   instantané ;
3. **Replay exact** : la simulation étant déterministe étant donné les poids et le stimulus,
   re-simuler sous les poids courants *est* le « exact current-policy replay » de RLT — on
   reconstruit toute l'histoire (y compris l'état récurrent du décodeur) avant d'évaluer
   l'action, sans approximation d'importance sampling ;
4. Le **bruit d'humeur** actuel (score + bruit ∝ état du Tamagotchi) devient un terme
   d'exploration explicitement modélisé (température β asservie à l'énergie/humeur) —
   biologiquement, c'est l'octopamine qui module le gain, on ne fait que le rendre
   cohérent dans la fonction d'objectif.

### B5. Co-design matériel : rien à changer, tout à défendre

Notre infrastructure répond déjà au principe 2 de RLT (encodage parallèle/état récurrent) :
batch de candidats en une matrice sparse, kernel C vérifié bit-exact, CSR > COO ×16, int8
symétrique validé. Les conclusions de nos benchmarks (SpearVM, HTC-Core, `bench_native.py`)
restent la garde-fou : **tout nouveau kernel du décodeur devra battre torch sur nos tailles
réelles** (le décodeur B2 est ~0,1 M FLOPs/pas — négligeable, aucun risque de régresser).

---

## 4. Plan d'implémentation priorisé

| # | Tâche | Fichiers | Effort | Dépend | Critère de réussite |
|---|---|---|---|---|---|
| 1 | Export annotations papier vers npz + masques | `prepare_data.py`, `data_loader.py` | ½ j | — | npz rechargeable, 1 420/948 retrouvés |
| 2 | Canaux sensoriels modaux (A2) | `encoding.py` | 1–2 j | 1 | réflexe Shiu intact ; variance inter-positions de la réponse ↑ (test SVD rank > 3) |
| 3 | Historique de lecture dans `brain.run` (A3.3) | `brain.py` | ¼ j | — | tenseur (T, n_read) sans régression perf |
| 4 | `LoopedReadout` + entraînement BPTT (B2) | `readout_looped.py`, `train_readout_looped.py` | 2–3 j | 3 | val ρ > 0,3 (n=25, 3 seeds) sinon abandon documenté |
| 5 | Théâtre : hotspots dimorphes (A4) | `live.py`, `connectome.js`, `index.html` | 1 j | 1 | grappes fru visibles à l'écran |
| 6 | Comportements dimorphes Tamagotchi (A4) | `tamagotchi.py`, `main.py` | 1 j | 1, 2 | « chanter » déclenché par canal olfactif |
| 7 | Readout restreint aux DN max-flow (A3) | `readout.py`/`readout_looped.py` | ½ j | 1, 4 | ablation DN-bruts vs DN-sélectionnés |
| 8 | Simulation différentiable (B3) | `brain.py` (branche) | 3–5 j | 4 | ρ ↑ vs B2 seul |
| 9 | Politique RL + replay exact (B4) | `chess_engine.py`, `tamagotchi.py` | 2–3 j | 4 | taux de victoire vs version max-score |

Ordre recommandé : **1 → 2 → 3 → 4 → 5** (le chemin critique vise le problème rank-1),
puis 6/7 en parallèle, 8/9 selon les résultats de 4.

## 5. Risques

1. **Le rank-1 peut persister** : si la dynamique LIF ne code rien, aucun décodeur ne l'inventera
   (B2 est conçu pour le *démontrer* aussi bien que pour le corriger — c'est une expérience,
   pas une promesse). Le levier suivant est alors le régime synaptique (dépression à court
   terme, cf. README) plutôt que le décodeur.
2. **Surapprentissage Stockfish** : le readout peut apprendre les features triviales
   (matériel) et ignorer le cerveau — garder l'ablation « readout sur features brutes vs
   lecture neurale » comme témoin négatif permanent.
3. **Colonnes d'annotations ≠ vérité terrain** : `potentially male-specific` (162 neurones)
   doit rester une catégorie à part dans toute visualisation.
4. **RLT non validé** : toutes les idées RLT sont derrière des étapes à critères mesurables et
   un critère d'abandon ; on ne l'adopte pas par dogme.

## 6. Références

- Papier : *Sexual dimorphism in the complete Drosophila male CNS connectome*, Cell 189:18,
  P5504–5526.E15 (3 sept. 2026) — https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
- Données/landing page : https://male-cns.janelia.org/ (les annotations `dimorphism`/`fruDsx`
  sont déjà dans notre `body-annotations.feather` v1.0)
- RLT : https://github.com/yifanzhang-pro/recurrent-looped-tranformer (rapport EN/ZH + site)
- Shiu et al., Nature 2024 (réflexe d'alimentation, α-synapses) — déjà répliqué ici.
- Interne : README.md (benchmarks, limites du readout), `exp_feeding_reflex.py`,
  `train_readout.py`.

## Addendum — pilote looped 13/09/2026 (90 positions, 1 seed, encodeur classic, DN all)

Protocole B2 exécuté en régime prod alpha/100 pas (le plafond 40 pas datait du régime x20 : 5 spikes moteurs à 40 pas contre 865 à 100 pas, mesuré). Verdict : **inconclusif, pas d'abandon** — le critère d'abandon exige trajectoire + canaux modaux, non testés ici.

| Readout (même val n=25, même régime) | rho Spearman |
|---|---|
| linéaire entraîné (ancien régime x20) | +0,061 (le fix de gain a effacé l'ancien 0,14) |
| linéaire aléatoire | +0,002 |
| looped pilote 90 pos | +0,105 (best train +0,156) |

Le looped bat le hasard mais reste dans le bruit (±0,2). Prochain run : --encoder modal --dn maxflow, ~500 pos, 3 seeds (~2 h/seed).
