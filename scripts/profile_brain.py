"""Bench bout-en-bout du run alpha sur MaleCNS : s/coup selon B et threads.
Usage : python scripts/profile_brain.py  (~5 min, comparatif chiffré)
"""
import sys
import time
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from data_loader import load_connectome
from brain import build_prod_brain
from encoding import BoardEncoder

conn = load_connectome()
enc = BoardEncoder(int(conn["is_sensory"].sum()))
boards8 = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")] * 4

for threads in (2, 4):
    torch.set_num_threads(threads)
    brain = build_prod_brain(conn)  # reconsruit après changement de threads
    for B in (1, 4, 8):
        cur = enc.encode_batch(boards8[:B]).to(brain.device)
        brain.run(cur, n_steps=10, record_every=10)  # warmup
        t = time.perf_counter()
        r = brain.run(cur, n_steps=100, record_every=10, return_history=True)
        dt = time.perf_counter() - t
        mot = float(r["motor_mean"].sum())
        print(f"[bench] threads={threads} B={B}: {dt:.1f}s / 100 pas "
              f"({dt:.0f}ms/pas, {dt / B:.1f}s par candidat, mot={mot:.0f})", flush=True)
