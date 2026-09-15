# Symbol Codex — registre des symboles → circuits

Chaque symbole est encodé comme topologie de circuit LIF (`backend/native/sigil_circuit.c`),
benché (`backend/bench_sigil.py` → `metrics.json` ci-contre) : taux, burstiness
(var/moy), fréquence dominante, capacité mémoire (MC, réservoir 20 délais).

Légende statut : ✅ implémenté + benché · 🔲 planifié · ❌ écarté (motif + raison).

## Runes — Futhark ancien (24, `futhark` variant = seed%40 → 0..23)

| # | Rune | Nom | Sens | Traits 1D encodés |
|---|---|---|---|---|
| 0 | ᚠ | Fehu | bétail/richesse | bâton + 2 brindilles hautes |
| 1 | ᚢ | Uruz | aurochs/force | bâton + diagonale de jonction |
| 2 | ᚦ | Thurisaz | géant/épine | bâton + triangle |
| 3 | ᚨ | Ansuz | dieu/bouche | bâton + 2 brindilles basses |
| 4 | ᚱ | Raidho | chevauchée/voyage | bâton + bol + jambe |
| 5 | ᚲ | Kenaz | torche | angle < (sans bâton) |
| 6 | ᚷ | Gebo | don | X (sans bâton) |
| 7 | ᚹ | Wunjo | joie | bâton + bol pointu |
| 8 | ᚺ | Hagalaz | grêle | bâton + parallèle + barre |
| 9 | ᚾ | Nauthiz | besoin | bâton + X médian |
| 10 | ᛁ | Isa | glace | bâton nu |
| 11 | ᛃ | Jera | année/récolte | deux chevrons convergents |
| 12 | ᛇ | Eihwaz | if | bâton + zigzag |
| 13 | ᛈ | Perthro | coupe/sort | bâton + bol rond |
| 14 | ᛉ | Algiz | élan/protection | bâton + paire haute |
| 15 | ᛊ | Sowilo | soleil | éclair (sans bâton) |
| 16 | ᛏ | Tiwaz | Tyr/justice | bâton + pointe de flèche |
| 17 | ᛒ | Berkano | bouleau | bâton + double bol |
| 18 | ᛖ | Ehwaz | cheval | bâton + diagonale longue |
| 19 | ᛗ | Mannaz | homme | bâton + X sommital |
| 20 | ᛚ | Laguz | eau | bâton + une brindille basse |
| 21 | ᛜ | Ingwaz | Ing/fertilité | losange (sans bâton) |
| 22 | ᛞ | Dagaz | jour | nœud papillon (2-cycle long) |
| 23 | ᛟ | Othala | héritage/domaine | losange + jambes |

## Runes — Futhark jeune (16, `futhark` variant 24..39, âge viking)

| # | Rune | Nom | Note |
|---|---|---|---|
| 24 | ᚠ | Fé | forme réduite (1 brindille) |
| 25 | ᚢ | Úr | crochet |
| 26 | ᚦ | Thurs | triangle ( glyph conservé du vieux) |
| 27 | ᚬ | Óss | brindille basse |
| 28 | ᚱ | Reið | bol (= perthro, conservé) |
| 29 | ᚴ | Kaun | brindille courte |
| 30 | ᚼ | Hagall | barre simple |
| 31 | ᚾ | Nauðr | X médian (conservé) |
| 32 | ᛁ | Íss | bâton nu (= isa, conservé) |
| 33 | ᛅ | Ár | diagonale |
| 34 | ᛋ | Sól | slash (sans bâton) |
| 35 | ᛏ | Týr | pointe de flèche (conservée) |
| 36 | ᛒ | Bjarkan | double bol (conservé) |
| 37 | ᛘ | Maðr | X sommital (conservé) |
| 38 | ᛚ | Lögr | brindille basse (conservée) |
| 39 | ᛦ | Ýr | paire haute (dérive d'algiz) |

Note 1D honnête : les côtés (gauche/droite) n'existent pas en projection 1D —
seules direction haut/bas et boucles survivent. Les codes identiques entre
ères (= thurisaz/thurs, isa/iss…) sont une parenté historique réelle, pas un bug.

## Runes — Futhorc anglo-saxon (`futhorc` variant = seed%33)

24 aînées conservées (mêmes glyphes, variant 0..23) + 9 propres (variant 24..32) :

| # | Rune | Nom | Traits encodés |
|---|---|---|---|
| 24 | ᚪ | Ac (chêne) | bâton + X latéral |
| 25 | ᚫ | Æsc (frêne) | bol + brindille haute |
| 26 | ᚣ | Yr (arc) | bâton + arc qui revient |
| 27 | ᛡ | Ior (anguille) | bâton + double zigzag |
| 28 | ᛠ | Ear (terre) | bâton + barre haute |
| 29 | ᛢ | Cweorth (feu) | bâton + diagonale longue |
| 30 | ᛣ | Calc (coupe) | bâton + coupe convergente |
| 31 | ᛥ | Stan (pierre) | bâton + losange (= othala + bâton) |
| 32 | ᚸ | Gar (lance) | bâton + double X |

## Staves islandais — `galdr` (variant = seed%4, Galdrabók et manuscrits)

| # | Stave | Motif encodé |
|---|---|---|
| 0 | Ægishjálmur (Heaume de Terreur) | moyeu + 8 bras à barbelures |
| 1 | Vegvísir (boussole) | 8 bras + anneau des pointes |
| 2 | Gapaldur | deux bâtons + barreaux |
| 3 | Ginfaxi | deux bâtons + X |

## Kabbale

| Symbole | Pattern | Note |
|---|---|---|
| Arbre de Vie (10 sephiroth + 22 sentiers, Kircher) | `tree` (n=10, 44 arêtes) | ✅ taille canonique imposée |
| 231 portes (Sefer Yetzirah 2:4 : C(22,2), explicite dans le texte) | `yetzirah` (n=22, K22, 462 arêtes) | ✅ validé par recherche web |

## Géométrie sacrée

| Symbole | Pattern | Note |
|---|---|---|
| Fleur de Vie | `flower` (treillis hexagonal torique) | ✅ top MC 0.26 |
| Cube de Métatron (13 cercles, Fruit de Vie) | `metatron` (n=13, K13, 156 arêtes) | ✅ |
| Sri Yantra (9 triangles entrelacés) | `sriyanta` (3 anneaux triangulaires + rayons) | ✅ |
| Vesica piscis | `vesica` (2 anneaux, 2 nœuds communs) | ✅ |
| Labyrinthe unicursal (Chartres : chemin hamiltonien) | `labyrinth` (serpent s×s) | ✅ |
| Pulli kolam (points + boucles fermées, sans bouts libres) | `kolam` (tore 8-connectivité) | ✅ validé par recherche |
| Mauvais œil / Nazar (anneaux concentriques) | `nazar` (3 anneaux + rayons) | ✅ |

## Sceaux (grimoires & sceaux classiques)

| Symbole | Pattern | Note |
|---|---|---|
| Sceau de Salomon (2 triangles entrelacés) | `hexagram` (cordes n/3) | ✅ |
| Pentagramme / pentacle (cercle + étoile) | `pentagram` / `pentacle` | ✅ |
| 72 sceaux goétiques (Lemegeton : anneau + glyphe interne) | `goetic` (anneau + 8 cordes, seed = démon 1..72) | ✅ famille paramétrique |
| Sigillum Dei Aemeth (anneau + heptagramme) | `enochian` (anneau + heptagramme + moyeu) | ✅ |
| Monas Hieroglyphica (Dee : soleil + croix + croissant) | `monas` | ✅ |
| Rose-Croix (croix latine + rose) | `rosecross` (croix + triangle central) | ✅ |
| Sceau de Baphomet | ❌ écarté : figuratif (tête de bouc), pas topologique |
| Carrés Abramelin (grilles de lettres palindromes) | `abramelin` (grille torique + accords miroirs sur 2 axes) | ✅ |

## Égypte

| Symbole | Pattern | Note |
|---|---|---|
| Ankh (vie : anneau + barre + queue) | `ankh` | ✅ |
| Œil d'Horus / Oudjat (sourcil + œil + spirale) | `wedjat` | ✅ |
| Scarabée (anneau + 6 pattes + suture) | `scarab` | ✅ pattes = sinks |
| Pilier Djed (stabilité : bâton + barres) | `djed` (bâton + 4 barres en tête) | ✅ |

## Nordique / germanique

| Symbole | Pattern | Note |
|---|---|---|
| Valknut (3 triangles entrelacés, nœud des tués) | `valknut` (3 anneaux triangulaires + triangle central) | ✅ |
| Mjölnir (marteau : manche + tête) | `mjolnir` (manche + tête triangulaire) | ✅ |
| Yggdrasil (frêne-monde : racines/tronc/cimes) | `yggdrasil` (DAG acyclique — premier motif sans cycle) | ✅ |
| Runes + staves | voir sections runes/galdr | ✅ |

## Celtique

| Symbole | Pattern | Note |
|---|---|---|
| Triquetra (3 anneaux en triangle) | `triquetra` | ✅ |
| Triskelion (3 bras récurrents) | `triskel` (moyeu + 3 bras) | ✅ |
| Nœud bouclier (4 anneaux carrés) | `shieldknot` | ✅ |
| Croix de Brigid (4 bras + centre tissé) | `brigid` (4 bras + carré central) | ✅ |
| Ogham (bâton/druim + rameaux, alphabet) | `ogham` (bâton + culs-de-sac : dissipation) | ✅ |

## Grèce / Rome / Mésopotamie

| Symbole | Pattern | Note |
|---|---|---|
| Labyrinthe (unicursal) | `labyrinth` | ✅ (voir géométrie) |
| Caducée (bâton + 2 serpents + ailes) | `caduceus` (bâton + serpents jumeaux + barre) | ✅ |

## Inde / Chine / diasporas

| Symbole | Pattern | Note |
|---|---|---|
| Sri Yantra | `sriyanta` | ✅ |
| Om (boucle + arc + croissant + point) | `om` (boucle + arc + croissant ; impression assumée) | ✅ |
| Yi Jing (64 hexagrammes = hypercube Q6, validé web) | `iching` (n=64, 384 arêtes) | ✅ |
| Bagua (8 trigrammes = cube Q3) | `bagua` (n=8, 24 arêtes) | ✅ |
| Taijitu (2 anneaux + épine S + 2 yeux) | `taijitu` | ✅ |
| Vèvè (carrefour en croix + symétrie miroir, validé web) | `veve` (croix + anneau + paires miroir) | ✅ |
| Adinkra nkyinkyim (torsades) | `adinkra` (zigzag + torsades longues) | ✅ |
| Hamsa (main : paume + 5 doigts + œil) | `hamsa` | ✅ |

## Alchimie / magie moderne

| Symbole | Pattern | Note |
|---|---|---|
| 4 éléments (triangles △▽ + barres) | `alchemy` (chaîne de triangles) | ✅ |
| 7 sceaux planétaires d'Agrippa (anneau + moyeu + k cordes, k = 3+planète, variant = seed%7 : Saturne..Lune) | `planetary` | ✅ |
| Sceaux planétaires (Agrippa : 7 sceaux) | 🔲 planifié (famille : anneau + k cordes, k = planète) |
| Sigils du chaos (intention → glyphe unique) | `chaosigil` (n=16, seed = hash d'intention) | ✅ |
| Ouroboros (serpent qui se mord) | `ouroboros` (cycle + morsure) | ✅ |

## Ressources sources (recherche web de cette session)
- Yi Dynamics AI (Q6, 384 arêtes) + iching-sheaf (spectre Q6, King Wen non-hamiltonien)
- Sefaria Sefer Yetzirah 2:2 + Codex Numerica (K22, 231 = C(22,2) explicite)
- Britannica + Wikipedia vèvè + analyse géométrique (carrefour + symétrie)
- Kolam : gating structure (2026), méthode topologique Forma, kambi kolam algorithmique
- Agent-reach skill chargé (recherche via search session pour ce savoir général)
