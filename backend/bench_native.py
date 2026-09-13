"""
Benchmark : backend LIF natif (kernel C) vs torch sparse CSR.
Workload réel de la mouche : n=5000, ~113k synapses, batch=20.
Vérifications :
  - C vs référence C naïve : 1M réseaux bit-exact (lif_verify.exe)
  - C vs torch : 300 réseaux aléatoires (native_lif.verify_vs_torch)
Usage : python bench_native.py
"""
import time

import numpy as np
import torch

torch.set_num_threads(2)

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
from native_lif import verify_vs_torch
import chess


def bench_step(brain, I_ext, reps=5, inner=100):
    best = 1e9
    for _ in range(reps):
        brain.reset(I_ext.shape[1])
        brain.step(I_ext)
        t0 = time.perf_counter()
        for _ in range(inner):
            brain.step(I_ext)
        best = min(best, (time.perf_counter() - t0) / inner * 1000)
    return best


def main():
    print("=== Vérification croisée C vs torch (300 réseaux aléatoires) ===")
    failures, trials = verify_vs_torch(300)
    status = "OK" if failures == 0 else "ÉCHEC"
    print(f"  {trials - failures}/{trials} passent, {failures} échecs -> {status}")
    if failures:
        return

    print("\n=== Benchmark pas LIF, n=5000, batch=20 ===")
    conn = load_connectome()
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"],
                     conn["is_descending"])
    encoder = BoardEncoder(n_sensory=brain.n_sensory)
    board = chess.Board()
    board.push_uci("e2e4")
    I_ext = (encoder.encode(board) * 20.0)[:, None].repeat(1, 20).contiguous()

    backend = "C natif" if brain._native is not None else "torch (natif indispo)"
    t_backend = bench_step(brain, I_ext)
    print(f"  {backend:<12} : {t_backend:8.3f} ms/pas")

    # Forcer le chemin torch pour comparaison
    brain_torch = FlyBrain.__new__(FlyBrain)
    brain_torch.__dict__.update(brain.__dict__)
    brain_torch._native = None
    t_torch = bench_step(brain_torch, I_ext)
    print(f"  {'torch CSR' :<12} : {t_torch:8.3f} ms/pas")
    print(f"  -> gain backend natif : ×{t_torch / t_backend:.2f}")

    print("\n=== Simulation complète run(300 pas), batch=20 ===")
    for name, br in [("C natif", brain), ("torch CSR", brain_torch)]:
        br.reset(20)
        t0 = time.perf_counter()
        br.run(I_ext, n_steps=300, record_every=10)
        print(f"  {name:<12} : {(time.perf_counter() - t0) * 1000:8.1f} ms")

    print("\n=== Équivalence dynamique : même run, les deux backends ===")
    brain.reset(20)
    r_native = brain.run(I_ext, n_steps=300, record_every=10)["motor_mean"]
    brain_torch.reset(20)
    r_torch = brain_torch.run(I_ext, n_steps=300, record_every=10)["motor_mean"]
    d = float((r_native - r_torch).abs().max())
    print(f"  écart max activité motrice : {d:.2e} -> "
          f"{'IDENTIQUE (tolérance 1e-4)' if d < 1e-4 else 'DIVERGENCE'}")


if __name__ == "__main__":
    main()
