"""
Expérience HTC-Core appliquée au connectome de la mouche.

Le papier HTC-Core quantifie les poids en ternaire {-1, 0, +1} (1.58 bit).
Questions testées ici sur le VRAI pipeline de la mouche :
  1. Mémoire : le format bitmask HTC compresse-t-il notre connectome ?
     (indice : notre matrice est SPARSE — le test est défavorable, et c'est
      instructif de mesurer à partir de quelle densité HTC gagne)
  2. Dynamique : que perd la simulation LIF si on binarise les poids ?
     corrélation de l'activité motrice baseline vs ternaire vs ternaire-scalé
  3. Vitesse : estimation du coût d'un pas LIF HTC (extraite du bench C
     375.6 ns/rangée de 64) vs torch sparse CSR.

Usage : python bench_htc_fly.py
"""
import time

import numpy as np
import torch

torch.set_num_threads(2)  # évite la contention OpenBLAS/torch sur 4 cœurs

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
import chess


def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_motor(brain, encoder, board):
    # Gain x20 calibré : en dessous (~x5), l'activité ne propage pas jusqu'aux
    # neurones moteurs et l'expérience serait vide.
    currents = encoder.encode(board).to(brain.device) * 20.0
    result = brain.run(currents, n_steps=300, record_every=10)
    return result["motor_mean"].squeeze(0).cpu().numpy()


def main():
    conn = load_connectome()
    W = conn["W"].tocsr()
    n = W.shape[0]
    nnz = W.nnz

    print("=" * 78)
    print("1. MÉMOIRE : bitmask HTC (2 bit/poids dense) vs notre format sparse")
    print("=" * 78)
    mem_sparse = (nnz * 4) + (nnz * 4) + n + 1          # data f32 + indices i32 + indptr
    mem_htc_dense = n * (n // 8) * 2                     # 2 bitmasks uint64/row, packed
    density = nnz / (n * n)
    print(f"   Connectome {n}x{n}, {nnz} synapses (densité {density*100:.2f}%)")
    print(f"   - Sparse (data+idx f32/i32)      : {mem_sparse/1e6:8.2f} Mo")
    print(f"   - HTC bitmasks denses (2 bit)    : {mem_htc_dense/1e6:8.2f} Mo")
    print(f"   -> À cette densité, HTC {'GAGNE' if mem_htc_dense < mem_sparse else 'PERD'} en mémoire.")
    # Densité de rupture : sparse <= bitmasks quand 8*nnz*(4+4+~4) <= n*n/4
    bytes_per_syn = 12.0
    break_even = (n * n / 4) / bytes_per_syn
    print(f"   - Rupture mémoire vers ~{break_even/1e6:.0f}M synapses (BitNet dense : oui ;")
    print(f"     connectomes biologiques à 0.2% : non — le sparse reste roi)")
    print(f"   - Pour info : MaleCNS réel ~50M synapses -> sparse ~400 Mo vs")
    print(f"     bitmasks ~6.9 Go : HTC perd x17 sur du câblage biologique creux.")

    print()
    print("=" * 78)
    print("2. VITESSE : pas LIF complet (n=5000, batch=20), mesuré AVANT les")
    print("   simulations lourdes de la section 3 (bruit de threads numpy/torch)")
    print("=" * 78)
    brain = FlyBrain(W, conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    encoder = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    I_ext = torch.zeros(brain.n_sensory, 20)
    best = 1e9
    for _ in range(5):
        brain.reset(20)
        brain.step(I_ext)
        t0 = time.perf_counter()
        for _ in range(100):
            brain.step(I_ext)
        best = min(best, (time.perf_counter() - t0) / 100 * 1000)
    t_sparse = best

    # Estimation HTC : bench C mesuré = 375.6 ns / rangée de 64 -> n rangées x batch
    t_htc = 375.6e-9 * n * 20 * 1000
    print(f"   - torch sparse CSR (actuel, meilleur des 5) : {t_sparse:8.3f} ms/pas")
    print(f"   - htc_compute (estimé depuis le C)          : {t_htc:8.3f} ms/pas  (x{t_htc/t_sparse:.0f})")
    print("   -> Sur CPU, HTC est un émulateur du silicium dédié ; le gain n'existe")
    print("      que sur FPGA/ASIC (0 DSP), hors de portée de ce projet.")

    print()
    print("=" * 78)
    print("3. DYNAMIQUE : simulation LIF baseline vs poids ternaires HTC")
    print("=" * 78)
    board = chess.Board()
    board.push_uci("e2e4")

    motor_base = run_motor(brain, encoder, board)

    # Variante A : ternaire pur {-1, 0, +1} (signe seulement)
    data_t = np.sign(W.data).astype(np.float32)
    W_ternary = torch.sparse_csr_tensor(
        torch.tensor(W.indptr, dtype=torch.int64),
        torch.tensor(W.indices, dtype=torch.int64),
        torch.tensor(data_t),
        size=W.shape,
    )
    brain_t = FlyBrain.__new__(FlyBrain)  # clone config, autre matrice
    brain_t.__dict__.update(brain.__dict__)
    brain_t.W = W_ternary
    motor_t = run_motor(brain_t, encoder, board)

    # Variante B : ternaire scalé (amplitude globale moyenne préservée)
    scale = float(np.abs(W.data).mean())
    W_scaled = torch.sparse_csr_tensor(
        torch.tensor(W.indptr, dtype=torch.int64),
        torch.tensor(W.indices, dtype=torch.int64),
        torch.tensor(data_t * scale, dtype=torch.float32),
        size=W.shape,
    )
    brain_s = FlyBrain.__new__(FlyBrain)
    brain_s.__dict__.update(brain.__dict__)
    brain_s.W = W_scaled
    motor_s = run_motor(brain_s, encoder, board)

    print(f"   Activité motrice moyenne : baseline {motor_base.mean():.4f} | "
          f"ternaire {motor_t.mean():.4f} | ternaire-scalé {motor_s.mean():.4f}")
    print(f"   Corrélation (Pearson) vecteurs moteurs vs baseline :")
    print(f"   - ternaire pur    : r = {pearson(motor_base, motor_t):+.3f}")
    print(f"   - ternaire scalé  : r = {pearson(motor_base, motor_s):+.3f}")
    print(f"   Erreur relative moyenne d'activité : "
          f"ternaire {np.abs(motor_t - motor_base).mean()/max(motor_base.mean(),1e-9)*100:.0f}% | "
          f"scalé {np.abs(motor_s - motor_base).mean()/max(motor_base.mean(),1e-9)*100:.0f}%")
    print("   -> Les poids synaptiques réels (loi gamma) ne sont PAS ternaires :")
    print("      quantifier détruit une partie du signal. HTC est exact pour des")
    print("      poids DÉJÀ ternaires (BitNet), pas pour du câblage biologique.")


if __name__ == "__main__":
    main()
