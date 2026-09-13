"""Accord du vainqueur à horizon réduit : 1 seule simu 100 pas / position,
scores recalculés sur préfixes d'historique (30/50/70/100).
Usage : python scripts/ablate_steps.py  (~10 min, 12 positions x 8 candidats)
"""
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from data_loader import load_connectome
from brain import build_prod_brain
from encoding import BoardEncoder
from readout import LinearReadout

conn = load_connectome()
brain = build_prod_brain(conn)
enc = BoardEncoder(int(conn["is_sensory"].sum()))
readout = LinearReadout(int(conn["is_motor"].sum()), int(conn["is_descending"].sum()))
st = torch.load("backend/readout.pt", map_location="cpu")
readout.load_state_dict(st, strict=False)
readout.eval()

# 12 positions variées (ouvertures + milieux), seed fixe
rng = np.random.default_rng(7)
positions = []
b = chess.Board()
positions.append(b.copy())
for _ in range(30):
    if b.is_game_over():
        break
    b.push(rng.choice(list(b.legal_moves)))
    if len(positions) < 12 and rng.random() < 0.5:
        positions.append(b.copy())

KS = [3, 5, 7, 10]  # frames -> 30/50/70/100 pas (record_every=10)
agree = {k: 0 for k in KS[:-1]}
n = 0
for pos in positions:
    cands = list(pos.legal_moves)[:8]
    if len(cands) < 3:
        continue
    boards = []
    for m in cands:
        pos.push(m)
        boards.append(pos.copy())
        pos.pop()
    cur = enc.encode_batch(boards).to(brain.device)
    r = brain.run(cur, n_steps=100, record_every=10, return_history=True)
    mh, dh = r["motor_hist"], r["descending_hist"]  # (10, B, .)
    winners = {}
    for k in KS:
        with torch.no_grad():
            sc = readout(mh[:k].mean(0), dh[k - 1]).cpu().numpy()
        winners[k] = int(np.argmax(sc))
    n += 1
    for k in KS[:-1]:
        agree[k] += winners[k] == winners[10]
    print(f"[steps] pos {n}: gagnants {winners}", flush=True)

print(f"ACCORD vainqueur vs 100 pas sur {n} positions : " +
      ", ".join(f"{k * 10}pas={agree[k] / n:.0%}" for k in KS[:-1]))
