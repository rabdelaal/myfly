# Rapport global — analyse de tout le système & usecases

Date : septembre 2026. Backend live (MaleCNS 183 490 neurones / 25 963 816 synapses).
Suite : 18/18 tests. Arbre git propre, Pages déployée.

## 1. Inventaire par couche

### 1.1 Cerveau + calcul natif
- MaleCNS réel, régime alpha prod (τ_syn 5 ms, délai 2, réfractaire 3).
- Kernel LIF α event-driven CSC (float32) : bit-exact 1M réseaux, parité torch exacte.
- Variante int16 (poids entiers exacts → 2× moins de trafic), bit-exact 100k.
- Kernel GEMV ternaire HTC-Core v2.1 porté : bit-exact 160k (dont −128),
  24.1/11.1 ns par bloc sur i7 (ratio 2.2× conforme au papier).
- SpearVM/superspear : gelu ×4.25, sigmoid ×2.03 branchés (mesurés, pas crus).

### 1.2 Mesure : leaderboard 7 domaines + codex 56 motifs
- Domaines : chess_reflex (en réparation), feeding 127–130 Hz, discrimination
  0.70–0.81, dynamics 0.70, speed −51/−71 ms, metaphor (stub), ie_state (stub).
- SymbolCodex : 56 topologies (sceaux, Kabbale Q22/K22, runes 40+33+4,
  Yi Jing Q6 validé web, vèvè, kolam…) + metrics.json auto-généré.
- Faits : flower MC 0.26 > random 0.06 (4×) ; denses ≈ 0.06 (la densité tue
  la mémoire) ; DAG/lignes gagnent le rappel-après-révision (E6).

### 1.3 Mémoire & apprentissage web (boucle fermée, live)
- search/learn (Exa ou DuckDuckGo) → /api/learn → /api/knowledge →
  /api/search (FlyHash 4/4, FP ~théorie) → pet study (cite) + Study.
- Digest extractif one-shot + endpooints reservoir / ie/generate / sigils.

### 1.4 Metabrain (Phase 0)
- AnythingEncoder (régime 450, md5 stable inter-processus), metaphor stub,
  ie_worlds (ledger+roster, replay déterministe), ie_state (vérité vs piège).

### 1.5 Reproductions indépendantes (preuves de rigueur)
- Collatz-Fibonacci : 2 cycles, 94.8/99.0 %, Th1/Th2 exhaustifs, λ=0.8774.
- Scrambling quantique N=8 : OTOC→0.93, Page 2.24/2.27, SFF plateau 1/256,
  GJW D=0.075 vs 1e-15, MSS λ/2πT≈0.1–0.19, rampe SFF pente 1.10 (R² 0.81).

### 1.6 Usecases rapides (5 protos vérifiés)
reservoir_kit, flyhash, watch_digest, sigil_music (MIDI), cpg_demo (allure
émergente mesurée, pas postulée).

## 2. Analyse honnête : forces, limites, résultats contre-intuitifs

Forces : tout est mesuré (jamais de claim sans bench), vérifié bit-exact,
testé (18/18), commité/poussé, avec boucles qui tournent seules (fine-tune,
watcher).
Limites connues : MC in-sample (surapprise, lire relative) ; benchs sous
contention CPU (fine-tune) ; régime réservoir input-dominé (E1) ; trigrammes
faibles sans embeddings ; transferts metabrain non prouvés (attendent teacher) ;
DuckDuckGo parfois vide (tests tolérants).
Contre-intuitif prouvé : l'oubli bat la mémoire sous révisions (E6) ; la
densité tue la mémoire ; le gel exact sans couplage (théorème GJW) ; −128
exact sans clamp (HTC).

## 3. Usecases par readiness

**Prêts (démonstrables aujourd'hui)** : mémoire sémantique du pet ;
veille auditée ; sigils-as-a-service (56 motifs) ; générateur IE ;
vitrine Sigils sonore ; musique générative ; FlyHash petits corpus ;
CPG simulées ; reproductions (pédagogie/recherche).
**Débloqués par le teacher** (ρ +0.651, en cours) : chess_reflex rempli ;
essaim distillé multi-topologies ; readout ternaire HTC ; transfert
métaphorique mesuré ; commentateur neuronal.
**Paris recherche** : encodeur par tour (ordre IE) ; régime réservoir
optimal ; codex multi-seeds + MC held-out ; FlyHash + embeddings ;
OTOC N=10 / SFF β>0.

## 4. Prochaines actions recommandées (ordre)
1. Laisser finir le teacher → re-benchmark → Phase 1 essaim (fil rouge).
2. Cron de veille + digest (1 h, produit immédiat).
3. Bench int16 ABANDONNÉ (zérote 99 % des poids normalisés) → remplacé par : eval rho n=25–50 + tirages moyennés (voir addendum).

## Addendum 16/09/2026
- Natif α vivant : 1,7 ms/pas (×33 vs torch 56) ; int16 abandonnée.
- Bench déterministe run-to-run ; référence rho looped ≈ −0,1 stable (3×15).
- Teacher mort à 160/200 (~14 h), non relancé : politique « jamais de train long » (budget 120 positions, `FLY_ALLOW_LONG_TRAIN=1`), `--resume-from` dispo, probe 120 pos = 1,1 min.
- `verify_vs_torch` : 299/300 sur cet env (torch 2.9.1 vs pin 2.5.0, à re-mesurer).
4. Codex multi-seeds + MC held-out (crédibilité).
5. Encodeur par tour → ie_state devient un vrai test d'ordre.
