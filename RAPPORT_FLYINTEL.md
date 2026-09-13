# FlyIntel — le cerveau de la mouche en module réutilisable

Le connectome MaleCNS (183 490 neurones, 25 963 816 synapses) est le cœur de
FlyChess. Ce dossier en fait un **module Python autonome et entraînable**,
détaché du serveur web, + un **leaderboard multi-domaines** + l'intégration
des kernels accélérés **SpearVM** et des formules **superspear**.

## TL;DR des résultats (sur MaleCNS réel, CPU)

| Domaine | Score | Détail |
|---|---|---|
| feeding | **127–130 Hz** | Réflexe MN9 sous drive GRN sucrée 100 Hz (Shiu) — le vrai cerveau répond |
| discrimination | **0.70–0.81** | log1p(inter/intra) des réponses motrices à 4 stimuli d'action |
| dynamics | **0.70** | cerveau vivant sans avalanche (sensoriel 9 Hz · moteur 1.4 Hz) |
| speed | **−51 à −71 ms/step** | ~14 pas/s sur MaleCNS CPU — le goulot (torch.sparse.mm) |
| chess_reflex | None | le readout actuel ne discrimine pas le val-set (à ré-entraîner) |

## Architecture

```
flyintel/
  __init__.py        API publique : load_brain, load_readout_module, ...
  backends.py        dispatch torch/SpearVM mesuré sur la machine (jamais les
                     chiffres du README) : gelu_fast, tanh_fast, benchmark_backends
  spear_math.py      ports Python des formules champions superspear (fast_exp,
                     gelu_spear, sigmoid_fast, aces, fresnel, ...) + benchmark
  readouts_spear.py  readouts (linéaire + looped) à activations Spear, entraînables
  train.py           wrapper d'entraînement (réutilise backend/train_readout_looped)
  bench.py           les 5 domaines + fusion leaderboard.json
benchmark_fly.py     CLI : benchmark réel / synthétique / --list
benchmarks/          leaderboard.json
tests/test_flyintel.py
```

Tout réutilise `backend/{brain,readout,readout_looped,encoding,data_loader}.py`
sans les dupliquer (ponytail : reuse, pas rewrite). `flyintel/__init__.py` ajoute
le chemin backend au `sys.path` et ré-exporte.

## Intégration SpearVM (kernels AVX2) + superspear (formules)

Mesures réelles sur **ce CPU** (i7-7660U, AVX2+FMA, torch CPU) :

| Opération | torch/libm | SpearVM/spear | Verdict |
|---|---|---|---|
| gelu (élémentaire, 1M) | 1.68 ms | **1.00 ms** | **×1.68** — dispatch SpearVM |
| tanh (élémentaire) | 9.48 ms | 122 ms | **×0.08** — torch gagne (rester torch) |
| matmul_nt_gelu (FFN 1024×768×3072) | 613 ms | 6113 ms | **×0.10** — MKL gagne massivement |
| gelu_spear algébrique | 24.77 ms | **5.82 ms** | **×4.25** — dispatch Spear |
| sigmoid_fast algébrique | 7.11 ms | **3.50 ms** | **×2.03** — dispatch Spear |
| fast_exp vs numpy.exp | 34.3 ms | 41.0 ms | **×0.84** — numpy AVX2 gagne |

**Conclusion honnête** (le superspear le martèle : *measure before shipping*) :
les formules **algébriques** (gelu_spear, sigmoid_fast) gagnent réellement et
sont branchées ; les approximations d'exponentielle perdent face à `numpy.exp`
vectorisé AVX2 ; le matmul reste à MKL. Les readouts Spear (`readouts_spear.py`)
sont entraînables (forward/backward + gradcheck ≤ 0.002).

## Entraînement réutilisable

```bash
python -m flyintel.train --dry-run                # fumée synthétique, rapide
python -m flyintel.train --episodes 200 --steps 50  # MaleCNS + stockfish (~2h)
python -c "from flyintel import train; train.train_looped(episodes=100, dry_run=True)"
```

Critère succès : val rho > +0.30 (n≥25) ; abandon si < +0.20 (le goulot est la
simulation LIF, pas le décodeur). Le readout actuel (`backend/readout.pt`)
donne rho dégénéré sur le val-set → à ré-entraîner avec le LoopedReadout.

## Leaderboard

```bash
python benchmark_fly.py --tag malecns-v1   # réel (déjà fait : voir benchmarks/)
python benchmark_fly.py --synthetic --tag demo
python benchmark_fly.py --list
```

## Limites / honnêteté

- **speed 71 ms/step** : le matmul sparse torch est le goulot sur CPU. Le vrai
  gain exigerait GPU ou un kernel natif event-driven (comme le CSC actif du
  worker frontend, qui fait ~15× moins de MAC à ~7 % d'activité).
- `chess_reflex` est vide sans readout entraîné qui discrimine (le rho dégénéré
  actuel est une vraie mesure, pas un bug).
- Les formules fast_exp/fog sont **bornées à leur bande** ([0,1], [0,2]) — hors
  domaine elles dégradent ; ne pas les utiliser sans garde.

## Suite (recursive self-improvement)

1. **Fine-tune complet** : `python -m flyintel.train --episodes 200` (~2h) puis
   re-benchmark `--tag malecns-v1.1`.
2. **Connecter le MCP superspear** (`discover`) pour découvrir de nouvelles
   formules pour les ops du readout / du LIF à la volée, puis les ré-injecter
   dans `spear_math.py` (boucle grounded-loop : c'est le "genuine recursive
   self-improvement").
3. **Kernel natif event-driven** pour le LIF (port du CSC actif du worker) :
   seul vrai levier pour passer sous les ~10 ms/step sur MaleCNS CPU.