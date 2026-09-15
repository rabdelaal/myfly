"""
Scrambling quantique & holographie (reproduction calculatoire) :
chaîne d'Ising en champ mixte (chaotique), N=8 spins, dim H = 256,
diagonalisation exacte + évolution unitaire.

Panneaux :
  P1 OTOC : C(t) = 1 - Re Tr[W(t) V W(t) V]/d (croissance + saturation ~1)
  P2 Page : entropie de von Neumann de 4 spins depuis Néel -> limite de
            Page ln(16) - 16/32 = 2.2726 nats
  P3 SFF  : K(t) = |Tr e^{-(b+it)H}|²/Z² (dip / rampe / plateau)
  P4 GJW  : teleportation TFD + couplage double-trace (g=0 vs g>0)

Usage : python -B quantum_scram.py [--quick]
"""
import sys

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def op_at(P, site, N):
    """Pauli P au site `site` (0 = le plus à gauche), identité ailleurs."""
    M = np.array([[1]], dtype=complex)
    for i in range(N):
        M = np.kron(M, P if i == site else I2)
    return M


def ising_hamiltonian(N=8, J=1.0, hx=1.05, hz=0.5):
    """Ising champ mixte, chaîne ouverte — point chaotique standard."""
    d = 2 ** N
    H = np.zeros((d, d), dtype=complex)
    for i in range(N - 1):
        H += -J * op_at(Z, i, N) @ op_at(Z, i + 1, N)
    for i in range(N):
        H += -hx * op_at(X, i, N) - hz * op_at(Z, i, N)
    return H


def diag(H):
    E, U = np.linalg.eigh(H)
    return E, U


def evolve_op(O, E, U, t):
    """O(t) = e^{iHt} O e^{-iHt} via la base propre."""
    ph = np.exp(1j * E * t)
    Oe = (U.conj().T @ O @ U) * ph[:, None]
    return U @ (Oe * ph[None, :].conj()) @ U.conj().T


def otoc(E, U, W, V, ts):
    d = U.shape[0]
    out = []
    for t in ts:
        Wt = evolve_op(W, E, U, t)
        F = np.trace(Wt @ V @ Wt @ V) / d
        out.append(float(1.0 - F.real))
    return np.array(out)


def otoc_thermal(E, U, W, V, beta, ts):
    """OTOC thermique : C(t) = 1 - Re Tr[rho W(t) V W(t) V], rho = e^{-bH}/Z.
    Tout en base propre (O(d^3) par pas) : WE, VE précalculés une fois."""
    p = np.exp(-beta * (E - E.min()))
    p /= p.sum()
    WE = U.conj().T @ W @ U
    VE = U.conj().T @ V @ U
    dE = E[:, None] - E[None, :]
    out = []
    for t in ts:
        Wt = WE * np.exp(1j * t * dE)
        M = (p[:, None] * Wt) @ VE @ Wt @ VE
        out.append(float(1.0 - np.trace(M).real))
    return np.array(out)


def fit_lambda(ts, C, lo=0.08, hi=0.25):
    """Exposant de Lyapunov : fit log C = a + λt sur la montée initiale.
    Fenêtre [0.08, 0.25] : assez haute pour sortir du bruit, assez basse
    pour rester avant saturation ET revivals (l'OTOC β=1 oscille après
    t~7, β=2 dès t~8 — fitter au-delà écrase la pente vers 0)."""
    m = (C > lo) & (C < hi)
    if m.sum() < 4:
        return float("nan"), float("nan"), 0
    x, y = ts[m], np.log(C[m])
    A = np.column_stack([x, np.ones_like(x)])
    sol, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    lam, n = sol[0], int(m.sum())
    err = float(np.sqrt(res[0] / max(n - 2, 1) / ((x ** 2).sum())) if len(res) else float("nan"))
    return float(lam), err, n


def sff_ramp_fit(E, beta=0.0, npts=200):
    """Ajuste la rampe SFF en log-log : logK = a + s·logt entre la fin du
    dip et 50% du plateau. À N=8 la rampe brute oscille sur un ordre de
    grandeur : on ajuste la MOYENNE GÉOMÉTRIQUE glissante (pratique standard,
    un seul Hamiltonien = pas de moyenne de désordre), et on reporte aussi
    le R² brut pour l'honnêteté."""
    tl = np.logspace(-1, 2.4, npts)
    K = sff(E, beta=beta, ts=tl)
    plat = K[-20:].mean()
    i_dip = int(np.argmin(K))
    i1 = int(np.searchsorted(K[i_dip:], plat * 0.5)) + i_dip
    i1 = max(i1, i_dip + 5)
    x, y = np.log(tl[i_dip:i1]), np.log(K[i_dip:i1])

    def _fit(xx, yy):
        A = np.column_stack([xx, np.ones_like(xx)])
        sol, res, _, _ = np.linalg.lstsq(A, yy, rcond=None)
        ss = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - float(res[0]) / ss if ss > 0 and len(res) else float("nan")
        return float(sol[0]), r2

    s_raw, r2_raw = _fit(x, y)
    w = 5  # moyenne géométrique glissante
    ys = np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])
    s_sm, r2_sm = _fit(x, ys)
    return (s_raw, r2_raw, s_sm, r2_sm, float(tl[i_dip]),
            float(tl[min(i1, len(tl) - 1)]))


def neel_state(N=8):
    v = np.zeros(2 ** N, dtype=complex)
    idx = sum((i % 2) * (2 ** (N - 1 - i)) for i in range(N))  # |0101...>
    v[idx] = 1.0
    return v


def yplus_state(N=8):
    """|+y>^N : énergie EXACTEMENT nulle (milieu du spectre -> T infinie).
    Néel a E=+7 (haut du spectre, T effective négative) : il thermalise
    sous Page par conservation de l'énergie, pas par défaut du chaos."""
    plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    v = np.array([1], dtype=complex)
    for _ in range(N):
        v = np.kron(v, plus)
    return v


def partial_trace_first(psi, N, keep):
    """Trace les N-keep derniers qubits, garde les `keep` premiers."""
    t = psi.reshape([2] * N)
    rho = np.tensordot(t, t.conj(), axes=(list(range(keep, N)),) * 2)
    return rho.reshape(2 ** keep, 2 ** keep)


def von_neumann(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return float(-(ev * np.log(ev)).sum())


def page_curve(E, U, N=8, keep=4, ts=None, init="yplus"):
    psi0 = yplus_state(N) if init == "yplus" else neel_state(N)
    c0 = U.conj().T @ psi0
    out = []
    for t in ts:
        psi = U @ (c0 * np.exp(-1j * E * t))
        out.append(von_neumann(partial_trace_first(psi, N, keep)))
    return np.array(out)


def sff(E, beta=0.5, ts=None):
    w = np.exp(-beta * E)
    Z = w.sum()
    num = np.abs(np.array([np.sum(w * np.exp(-1j * E * t)) for t in ts])) ** 2
    return num / Z ** 2


# --- P4 : teleportation Gao-Jafferis-Wall (version qubit minimale) ---
def tfd(E, beta=1.0):
    w = np.exp(-beta * E / 2)
    c = np.diag(w / np.sqrt((w ** 2).sum()))
    return c  # coeffs c[m,n] sur |m>_L |n>_R


def evolve2(c, E, t):
    return c * np.exp(-1j * t * (E[:, None] + E[None, :]))


def expect_R(c, OE):
    """<I_L ⊗ O_R> exact. Note : OE hermitienne => OE.conj().T == OE.
    (Une version antérieure utilisait OE.T, soit <O^T> — identique pour
    X/Z symétriques, signe inversé pour Y antisymétrique.)"""
    return complex(np.sum(c.conj() * (c @ OE.conj().T)))


# --- P4 : teleportation Gao-Jafferis-Wall (version qubit minimale) ---
# Observable rigoureuse : DISTINGUABILITÉ du message. Deux messages
# (insertion X_L vs Z_L), distance de trace entre états réduits du qubit R
# à la lecture. Sans couplage, L et R sont découplés : sorties identiques
# (transmission nulle). Avec couplage double-trace (trou traversable),
# les sorties diffèrent : le message a traversé. Théorème utile : sans
# couplage, <P_R> est EXACTEMENT gelé (l'unitarité côté L effondre la somme
# sur m via V†V=I), donc D(g=0) = 0 à la précision machine — fond nul.
def bloch_R(c, ops):
    return np.array([expect_R(c, O).real for O in ops])


def gjw_run(E, U, insert, beta, g, t0, t_read, site=0):
    NL = U.conj().T @ op_at(insert, site, 8) @ U
    ops = [U.conj().T @ op_at(P, site, 8) @ U for P in (X, Y, Z)]
    ZL = U.conj().T @ op_at(Z, site, 8) @ U
    ZR = ops[2]
    c = NL @ tfd(E, beta)
    c = evolve2(c, E, t0)
    cg, sg = np.cos(g), np.sin(g)  # e^{igV} = cos g + i sin g V (V²=1)
    c = cg * c + 1j * sg * (ZL @ c @ ZR.T)
    cc = evolve2(c, E, t_read)
    n = np.sqrt(np.sum(np.abs(cc) ** 2))
    return bloch_R(cc / n, ops)


def gjw(E, U, beta=1.0, g=1.0, t0=4.0, t_read=None, site=0):
    """Contraste de teleportation : distance de trace entre sorties R pour
    messages X vs Z, avec et sans couplage. Retourne (D_sans, D_avec)."""
    tr = t0 if t_read is None else t_read
    rX0 = gjw_run(E, U, X, beta, 0.0, t0, tr, site)
    rZ0 = gjw_run(E, U, Z, beta, 0.0, t0, tr, site)
    rX1 = gjw_run(E, U, X, beta, g, t0, tr, site)
    rZ1 = gjw_run(E, U, Z, beta, g, t0, tr, site)
    D0 = float(np.linalg.norm(rX0 - rZ0) / 2)
    D1 = float(np.linalg.norm(rX1 - rZ1) / 2)
    return D0, D1


def main(quick=False):
    N = 8
    print("diag exacte : chaîne Ising champ mixte, N=8, dim =", 2 ** N)
    E, U = diag(ising_hamiltonian(N))
    print(f"  fondamental E0={E[0]:.4f}, largeur spectrale={E[-1]-E[0]:.2f}")

    ts = np.linspace(0, 12, 61)
    W = op_at(X, 0, N)
    V = op_at(Z, N - 1, N)
    C = otoc(E, U, W, V, ts)
    print(f"P1 OTOC : C(0)={C[0]:.4f} -> saturation ~{C[-10:].mean():.3f} "
          f"(attendu ~1) ; t(C=0.5)~{ts[np.searchsorted(C, 0.5)]:.2f}")

    S = page_curve(E, U, N, 4, ts)
    Spage = float(np.log(16) - 16 / 32)
    print(f"P2 Page : S(0)={S[0]:.4f} -> {S[-10:].mean():.4f} "
          f"(limite théorique {Spage:.4f})")

    tl = np.logspace(-1, 2.2, 120)
    K = sff(E, beta=0.0, ts=tl)
    late = K[-20:].mean()
    # plateau théorique β=0 : (1/d²)Σ_degen ; 1/256 sans dégénérescence.
    print(f"P3 SFF : dip {K.min():.2e} à t~{tl[K.argmin()]:.2f}, "
          f"plateau ~{late:.2e} (1/256={1/256:.2e}, x{late * 256:.1f} = "
          f"dégénérescence parité)")

    t0, beta, g, tr = 6.0, 1.0, 4.0, 8.0
    D0, D1 = gjw(E, U, beta, g, t0, tr)
    print(f"P4 GJW : distinguabilite messages X vs Z sur R : "
          f"sans couplage D={D0:.2e} (~0, trou non traversable) | "
          f"avec couplage D={D1:.3f} (message transmis)")
    if not quick:
        for tt in (2.0, 4.0, 6.0, 8.0, 10.0):
            a, b = gjw(E, U, beta, g, t0, tt)
            print(f"   t_read={tt}: D_sans={a:.2e} D_avec={b:.3f}")


def main_mss():
    """OTOC thermique + borne MSS (λ ≤ 2πT) et rampe SFF ajustée."""
    N = 8
    E, U = diag(ising_hamiltonian(N))
    W = op_at(X, 0, N)
    V = op_at(Z, N - 1, N)
    ts = np.linspace(0, 14, 141)
    print("OTOC thermique : fit log C = a + L*t ; borne MSS L <= 2*pi*T (J=1)")
    for beta in (0.5, 1.0, 2.0):
        C = otoc_thermal(E, U, W, V, beta, ts)
        lam, err, n = fit_lambda(ts, C)
        T = 1.0 / beta
        print(f"  beta={beta} (T={T:.1f}) : L={lam:.3f}+-{err:.3f} ({n} pts) | "
              f"2piT={2 * np.pi * T:.2f} | L/2piT={lam / (2 * np.pi * T):.3f} "
              f"| Cmax={C.max():.3f}")
    s_raw, r2_raw, s_sm, r2_sm, td, tp = sff_ramp_fit(E)
    print(f"SFF rampe log-log t=[{td:.2f},{tp:.2f}] : brute s={s_raw:.2f} "
          f"(R²={r2_raw:.2f}) | lissée s={s_sm:.2f} (R²={r2_sm:.3f}, attendu ~1)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mss":
        main_mss()
    else:
        main("--quick" in sys.argv)
