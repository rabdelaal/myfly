"""
Benchmark : baseline fly vs version spear (SpearVM AVX2 + champions).

Composants mesurés (tous présents dans le pipeline neural de la mouche) :
  1. Projection d'encodage (dense)      : numpy        vs spur matmul_nt
  2. Couche dense fusionnée matmul+gelu : numpy+scipy  vs spur matmul_nt_gelu
  3. tanh (4M éléments)                 : numpy        vs spur tanh (champion)
  4. Pas LIF sparse CSR (torch)         : baseline seule — aucun kernel sparse
                                          chez spear (gain asymétrique : rien à remplacer)
  5. Variante LIF dense (si W était dense) : torch dense vs spur matmul_nt

Chaque paire est validée numériquement (écart max) avant d'être chronométrée.
Usage : python bench_spear.py
"""
import os
import time

import numpy as np

try:
    import spur_math
    SPUR = True
    SPUR_ERR = None
except Exception as e:  # DLL absente, CPU sans AVX2, etc.
    SPUR = False
    SPUR_ERR = e

import torch
import scipy.special

from data_loader import load_connectome
from brain import FlyBrain


def bench(fn, reps=100, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)  # meilleur temps = mesure standard des kernels


def row(name, base_t, spear_t, err=None):
    if spear_t is None:
        print(f"  {name:<44} baseline {base_t*1e3:9.3f} ms | spear INDISPONIBLE")
        return
    gain = base_t / spear_t
    err_s = f" | écart max {err:.2e}" if err is not None else ""
    print(f"  {name:<44} baseline {base_t*1e3:9.3f} ms | spear {spear_t*1e3:9.3f} ms | ×{gain:6.2f}{err_s}")


def main():
    print(f"spur_math disponible : {SPUR}" + (f" ({SPUR_ERR})" if not SPUR else ""))
    rng = np.random.default_rng(0)

    # --- 1. Projection d'encodage (dense) ---
    print("\n[1] Projection d'encodage features → neurones sensoriels (fp32)")
    for n_sensory, label in [(1500, "synthétique (1500 sensoriels)"),
                             (50000, "échelle MaleCNS (50 000 sensoriels)")]:
        feats = rng.normal(size=(20, 788)).astype(np.float32)
        proj = rng.normal(scale=1.0 / 28.0, size=(788, n_sensory)).astype(np.float32)
        proj_t = np.ascontiguousarray(proj.T)
        t_np = bench(lambda: feats @ proj, reps=200 if n_sensory == 1500 else 20)
        if SPUR:
            out_np = feats @ proj
            out_sp = spur_math.matmul_nt(feats, proj_t)
            err = float(np.max(np.abs(out_np - out_sp)))
            t_sp = bench(lambda: spur_math.matmul_nt(feats, proj_t),
                         reps=200 if n_sensory == 1500 else 20)
            row(f"encode_batch 20 candidats, {label}", t_np, t_sp, err)
        else:
            row(f"encode_batch 20 candidats, {label}", t_np, None)

    # --- 2. Couche dense fusionnée (matmul + GELU) ---
    print("\n[2] Couche dense fusionnée matmul+GELU — readout non-linéaire (fp32)")
    x = rng.normal(size=(20, 788)).astype(np.float32)
    w = rng.normal(scale=0.02, size=(4096, 788)).astype(np.float32)
    b = rng.normal(size=(4096,)).astype(np.float32)

    def ffn_numpy():
        h = x @ w.T + b
        return 0.5 * h * (1.0 + scipy.special.erf(h / np.sqrt(2.0)))

    t_np = bench(ffn_numpy, reps=200)
    if SPUR:
        out_np = ffn_numpy()
        out_sp = spur_math.matmul_nt_gelu(x, w, b)
        err = float(np.max(np.abs(out_np - out_sp)))
        t_sp = bench(lambda: spur_math.matmul_nt_gelu(x, w, b), reps=200)
        row("FFN 20×788→4096 fusionnée", t_np, t_sp, err)
    else:
        row("FFN 20×788→4096 fusionnée", t_np, None)

    # --- 3. tanh sur 4M éléments (champion transcendental) ---
    print("\n[3] tanh, 4M éléments (champion SPEAR)")
    big = rng.normal(size=4_000_000).astype(np.float32)
    t_np = bench(lambda: np.tanh(big), reps=20)
    if SPUR:
        err = float(np.max(np.abs(np.tanh(big) - spur_math.tanh(big))))
        t_sp = bench(lambda: spur_math.tanh(big), reps=20)
        row("np.tanh vs spur tanh (4M)", t_np, t_sp, err)
    else:
        row("np.tanh vs spur tanh (4M)", t_np, None)

    # --- 4 & 5. Le cerveau : LIF sparse et variante dense ---
    print("\n[4] Cœur du cerveau : pas LIF sparse CSR, batch 20 (torch)")
    conn = load_connectome()
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc_in = rng.normal(size=(brain.n_sensory, 20)).astype(np.float32)
    I_ext = torch.tensor(enc_in)
    brain.reset(20)
    t_step = bench(lambda: brain.step(I_ext), reps=200, warmup=10)
    row("1 pas LIF sparse CSR (B=20)", t_step, None)  # pas d'équivalent spear

    print("\n[5] Variante hypothétique : connectome DENSE 5000×5000 (B=20)")
    Wd = torch.tensor(np.asarray(conn["W"].todense(), dtype=np.float32))
    spikes = torch.zeros(brain.n, 20)
    spikes[:50] = 1.0
    t_torch = bench(lambda: torch.mm(Wd, spikes), reps=100, warmup=5)
    if SPUR:
        Wd_np = Wd.numpy()
        sp_np = spikes.numpy()
        out_torch = (Wd @ spikes).numpy()
        out_spur = spur_math.matmul_nt(Wd_np, sp_np.T)
        err = float(np.max(np.abs(out_torch - out_spur)))
        t_sp = bench(lambda: spur_math.matmul_nt(Wd_np, np.ascontiguousarray(sp_np.T)),
                     reps=100, warmup=5)
        row("1 pas LIF dense : torch.mm vs spur matmul_nt", t_torch, t_sp, err)
    else:
        row("1 pas LIF dense : torch.mm", t_torch, None)

    print("\n--- Synthèse ---")
    print("Le gain spear est asymétrique : fort sur les ops denses et les")
    print("transcendantales, nul sur le sparse (le cerveau reste en torch CSR).")


if __name__ == "__main__":
    main()
