"""
Benchmark : backend LIF α event-driven (lif_alpha_ed.dll) vs torch sparse.mm
sur MaleCNS, batch=1 (régime prod live.py).

Objectif : battre les ~51-71 ms/step du sparse.mm dense en ne dispersant que
les neurones actifs (~7 %).

Vérifications :
  - C vs référence C naïve : 1M réseaux bit-exact (lif_alpha_ed_verify.exe)
  - C vs torch : 300 réseaux aléatoires (native_lif_alpha.verify_vs_torch)
  - Parité dynamique sur MaleCNS (même run, deux backends)
Usage : python bench_native_alpha.py
"""
import time

import numpy as np
import torch

torch.set_num_threads(2)

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
from native_lif_alpha import verify_vs_torch
import chess


def bench_step(brain, I_ext, reps=5, inner=100):
    best = 1e9
    for _ in range(reps):
        brain.reset(1)
        brain.step(I_ext)
        t0 = time.perf_counter()
        for _ in range(inner):
            brain.step(I_ext)
        best = min(best, (time.perf_counter() - t0) / inner * 1000)
    return best


def main():
    print("=== Vérification croisée C α vs torch (300 réseaux aléatoires) ===")
    failures, trials = verify_vs_torch(300)
    status = "OK" if failures == 0 else "ÉCHEC"
    print(f"  {trials - failures}/{trials} passent, {failures} échecs -> {status}")
    if failures > 5:
        return

    print("\n=== Benchmark pas α-LIF, MaleCNS réel, batch=1 ===")
    conn = load_connectome()
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"],
                     conn["is_descending"], synapse_model="alpha",
                     tau_syn=5.0, delay_steps=2, refractory_steps=3)
    encoder = BoardEncoder(n_sensory=brain.n_sensory)
    board = chess.Board()
    board.push_uci("e2e4")
    I_ext = (encoder.encode(board) * 20.0)[:, None]

    backend = "C event-driven" if brain._native_alpha is not None else "torch (natif indispo)"
    t_backend = bench_step(brain, I_ext)
    print(f"  {backend:<14} : {t_backend:8.3f} ms/pas")

    brain_torch = FlyBrain.__new__(FlyBrain)
    brain_torch.__dict__.update(brain.__dict__)
    brain_torch._native_alpha = None
    t_torch = bench_step(brain_torch, I_ext)
    print(f"  {'torch sparse' :<14} : {t_torch:8.3f} ms/pas")
    print(f"  -> gain event-driven : ×{t_torch / t_backend:.2f}")

    print("\n=== Parité dynamique sur MaleCNS (même run, deux backends) ===")
    brain.reset(1)
    r_native = brain.run(I_ext, n_steps=200, record_every=10)["motor_mean"]
    brain_torch.reset(1)
    r_torch = brain_torch.run(I_ext, n_steps=200, record_every=10)["motor_mean"]
    d = float((r_native - r_torch).abs().max())
    print(f"  écart max activité motrice : {d:.2e} -> "
          f"{'IDENTIQUE (tolérance 1e-4)' if d < 1e-4 else 'DIVERGENCE'}")


if __name__ == "__main__":
    main()