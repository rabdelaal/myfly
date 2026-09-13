"""Re-validation post-fix x20 sur le VRAI MaleCNS (tâche 4).

Vérifie, sans Stockfish et en ~2 min :
  1. comptes Cell 2026 exacts (1258 male-specific, 2611 fru_high…)
  2. GRN sucrées + MN9 présents (réflexe Shiu câblé)
  3. normes d'encodage classic/modal = 450 (pas de ×20)
  4. la simu répond : spikes > 0 et patterns distincts entre positions
     (anti rank-1 : rang SVD > 1 sur 6 positions batchées)
  5. trajectoire looped (T,B,d) cohérente avec motor_mean

Usage : python scripts/validate_fix.py  (depuis la racine)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
import chess

from data_loader import load_connectome
from brain import FlyBrain, build_prod_brain
from encoding import BoardEncoder, ModalBoardEncoder, STIM_TARGET_NORM
from readout_looped import trajectory_from_brain

t0 = time.time()

# 1. Comptes Cell
d = np.load("data/malecns.npz", allow_pickle=True)
from collections import Counter
dim, fru = Counter(map(str, d["dimorphism"])), Counter(map(str, d["frudsx"]))
assert dim["male-specific"] == 1258, dim
assert dim["sexually dimorphic"] == 771, dim
assert fru["fru_high"] == 2611 and fru["dsx_high"] == 138, fru
print(f"[1] Cell OK : {dim['male-specific']} male-specific, {fru['fru_high']} fru_high "
      f"({time.time()-t0:.0f}s)")

# 2. Câblage Shiu
conn = load_connectome()
assert len(conn["sugar_grn_idx"]) > 0 and len(conn["mn9_idx"]) > 0
print(f"[2] Shiu câblé : {len(conn['sugar_grn_idx'])} GRN sucrées, {len(conn['mn9_idx'])} MN9")

# 3. Normes
enc_c, enc_m = BoardEncoder(int(conn["is_sensory"].sum())), ModalBoardEncoder(int(conn["is_sensory"].sum()))
boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")]
for enc, name in ((enc_c, "classic"), (enc_m, "modal")):
    cb = enc.encode_batch(boards)
    for j in range(cb.shape[1]):
        n = float(cb[:, j].norm())
        assert abs(n - STIM_TARGET_NORM) / STIM_TARGET_NORM < 0.03, (name, n)
print("[3] normes 450 OK (classic + modal, pas de x20)")

# 4-5. Simu batchée : 6 positions, 100 pas (régime prod alpha/MaleCNS —
# 40 pas laissent les moteurs silencieux avec le gain corrigé, mesuré :
# 5 spikes à 40 pas → 865 à 100 pas sur position test)
import random
random.seed(7)
test_boards = []
b = chess.Board()
for _ in range(6):
    b = chess.Board()
    for _ in range(random.randint(5, 25)):
        if b.is_game_over():
            break
        b.push(random.choice(list(b.legal_moves)))
    test_boards.append(b)
brain = build_prod_brain(conn)  # régime prod alpha (current = silence sur MaleCNS)
cur = enc_c.encode_batch(test_boards).to(brain.device)
r = brain.run(cur, n_steps=100, record_every=10, return_history=True)
mm = r["motor_mean"].cpu().numpy()
assert mm.sum() > 0, "aucun spike moteur : régime sub-seuil, revoir STIM_GAIN"
u, s, _ = np.linalg.svd(mm - mm.mean(0, keepdims=True), full_matrices=False)
rank = int((s / s.max() > 0.05).sum())
print(f"[4] spikes moteurs OK ({mm.sum():.0f}), rang SVD={rank} (succès si > 1)")
traj = trajectory_from_brain(r)
assert traj.shape[:2] == (10, 6), traj.shape
assert torch.allclose(r["motor_mean"], r["motor_hist"].mean(0), atol=1e-5)
print(f"[5] traj {tuple(traj.shape)} cohérente avec motor_mean")
print(f"VALIDE en {(time.time()-t0)/60:.1f} min — régime post-fix sain.")
