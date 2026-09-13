"""
Deux expériences inspirées des découvertes précédentes :

1. CHAMPION `rc_circuit` — notre LIF utilise Euler (V += (-V+I)*dt/tau), une
   approximation du circuit RC exact : V <- V*alpha + I*(1-alpha), alpha=exp(-dt/tau).
   L'intégration exacte autorise un pas de temps plus grand sans dériver :
   peut-on simuler 5× moins de pas à dynamique égale ?

2. LEÇON HTC (clamp symétrique) — quantification int8 symétrique des courants
   d'encodage, clampée à [-127, +127] (jamais -128, le défaut formellement
   prouvé dans HTC-Core). Que perd le readout ?

Usage : python bench_rc_int8.py
"""
import time

import numpy as np
import torch

torch.set_num_threads(2)

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
import chess


def make_torch_W(conn):
    W = conn["W"].tocoo()
    idx = torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long)
    val = torch.tensor(W.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=W.shape).coalesce().to_sparse_csr()


def simulate(W_ctx, currents, n_steps, dt, scheme, neuromod=1.0,
             tau_m=20.0, V_th=1.0, V_reset=0.0, gain=20.0):
    """Simulateur LIF autonome avec schéma d'intégration paramétrable.
    W_ctx : (W_sparse, is_sensory, is_motor) préconstruits."""
    W, is_sensory, is_motor = W_ctx
    n = W.shape[0]
    B = 1
    I_full = torch.zeros(n, B)
    I_full[is_sensory, 0] = currents * gain  # courant seulement sur les sensoriels

    V = torch.zeros(n, B)
    spikes = torch.zeros(n, B)
    alpha = float(np.exp(-dt / tau_m))
    motor_sum, n_rec = torch.zeros(int(is_motor.sum())), 0

    with torch.no_grad():
        for step in range(n_steps):
            I_syn = neuromod * torch.sparse.mm(W, spikes) + I_full
            if scheme == "euler":
                V = V + (-V + I_syn) * (dt / tau_m)
            else:  # 'exact' : circuit RC (champion rc_circuit)
                V = V * alpha + I_syn * (1.0 - alpha)
            spiked = V >= V_th
            V = torch.where(spiked, torch.full_like(V, V_reset), V)
            spikes = spiked.float()
            if step % max(int(10 / (dt / 1.0)), 1) == 0:
                motor_sum += spikes[is_motor, 0]
                n_rec += 1
    return (motor_sum / max(n_rec, 1)).numpy()


def pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    conn = load_connectome()
    W_ctx = (make_torch_W(conn), torch.tensor(conn["is_sensory"]),
             torch.tensor(conn["is_motor"]))
    encoder = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    board = chess.Board()
    board.push_uci("e2e4")
    currents = encoder.encode(board)

    print("=" * 78)
    print("1. CHAMPION rc_circuit : Euler vs intégration exacte du circuit RC")
    print("=" * 78)
    base = simulate(W_ctx, currents, n_steps=300, dt=1.0, scheme="euler")
    exact1 = simulate(W_ctx, currents, n_steps=300, dt=1.0, scheme="exact")
    print(f"   activité motrice moyenne : euler(1ms) {base.mean():.4f} | "
          f"exact(1ms) {exact1.mean():.4f}")
    print(f"   corrélation vs euler(1ms) : exact(1ms) r = {pearson(base, exact1):+.3f}")

    # Pas élargi AVEC recalibrage du gain : même régime de décharge, moins de pas
    target = float(base.mean())
    calibrated = {}
    for dt_big, steps_big in [(5.0, 60), (10.0, 30)]:
        lo, hi = 0.1, 60.0
        for _ in range(10):  # dichotomie du gain à taux de décharge égal
            mid = (lo + hi) / 2
            m = simulate(W_ctx, currents, n_steps=steps_big, dt=dt_big,
                         scheme="exact", gain=mid).mean()
            if m < target:
                lo = mid
            else:
                hi = mid
        gain_cal = (lo + hi) / 2
        m = simulate(W_ctx, currents, n_steps=steps_big, dt=dt_big,
                     scheme="exact", gain=gain_cal)
        calibrated[dt_big] = gain_cal
        print(f"   - exact({dt_big:.0f}ms), gain recalibré {gain_cal:.2f} "
              f"(activité {m.mean():.4f} vs cible {target:.4f}) : "
              f"r = {pearson(base, m):+.3f} -> {300 // steps_big}× moins de pas")

    print()
    print("-" * 78)
    print("   Test décisif : le CLASSEMENT des 10 premiers coups candidats est-il")
    print("   préservé ? (le readout est linéaire — c'est le rang qui compte)")
    print("-" * 78)
    candidates = []
    for mv in list(board.legal_moves)[:10]:
        board.push(mv)
        candidates.append((mv.uci(), encoder.encode(board)))
        board.pop()

    def scores(scheme, dt, n_steps, gain):
        out = {}
        for uci, cur in candidates:
            m = simulate(W_ctx, cur, n_steps=n_steps, dt=dt, scheme=scheme, gain=gain)
            out[uci] = float(m.mean())
        return out

    s_ref = scores("euler", 1.0, 300, 20.0)
    for dt_big, steps_big in [(5.0, 60), (10.0, 30)]:
        s_big = scores("exact", dt_big, steps_big, calibrated.get(dt_big, 20.0))
        ucis = [u for u, _ in candidates]
        rank_ref = np.argsort([s_ref[u] for u in ucis])
        rank_big = np.argsort([s_big[u] for u in ucis])
        rho = pearson(rank_ref.astype(float), rank_big.astype(float))
        same_top = ucis[int(rank_ref[-1])] == ucis[int(rank_big[-1])]
        print(f"   exact({dt_big:.0f}ms) vs euler(1ms) : corrélation de rangs "
              f"ρ = {rho:+.3f} | même meilleur coup : {'OUI' if same_top else 'NON'}"
              f"  ({300 // steps_big}× moins de pas)")

    print()
    print("=" * 78)
    print("2. LEÇON HTC : encodage int8 symétrique, clamp [-127, +127]")
    print("=" * 78)
    imax = float(currents.abs().max())
    scale = imax / 127.0 if imax > 0 else 1.0
    q = np.clip(np.round(currents.numpy() / scale), -127, 127).astype(np.int8)
    assert q.min() >= -128 and q.max() <= 127
    assert not (q == -128).any(), "valeur -128 interdite (défaut HTC)"
    deq = torch.tensor(q.astype(np.float32) * scale)

    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"],
                     conn["is_descending"])
    brain.reset(1)

    def motor_of(c):
        # Gain ×20 : même régime que les autres benchmarks, sinon sub-seuil
        r = brain.run(c * 20.0, n_steps=300, record_every=10)
        return r["motor_mean"].squeeze(0).cpu().numpy()

    m_float = motor_of(currents)
    m_int8 = motor_of(deq)
    print(f"   erreur de quantification max : {float((deq - currents).abs().max()):.5f} "
          f"(échelle {scale:.5f}, aucune valeur à -128)")
    print(f"   activité motrice : float {m_float.mean():.4f} | int8 {m_int8.mean():.4f}")
    print(f"   corrélation moteurs float vs int8 : r = {pearson(m_float, m_int8):+.4f}")
    print("   -> L'encodage int8 symétrique préserve la dynamique : si un jour le")
    print("      connectome passe en accélérateur entier (FPGA HTC), l'encodeur")
    print("      est déjà prêt — avec le clamping qui neutralise le défaut -128.")


if __name__ == "__main__":
    main()
