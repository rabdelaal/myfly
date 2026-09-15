"""
Bench : circuits virtuels sigillaires — vitesse + propriétés émergentes.

Pour chaque pattern (seal, pentagram, ring, wheel, grid, ziggurat, random) :
  - génère l'edge-list via le C (sigil_circuit.dll)
  - assemble W (+w / 15 % inhibiteurs), normalise le rayon spectral à 0.9
    pour que seule la TOPOLOGIE pilote les différences émergentes
  - drive binaire aléatoire identique pour tous (même seed → comparable)
  - mesure : pas/s (C dense), taux moyen, burstiness (var/moy du popcount),
    fréquence dominante (FFT du taux de population), capacité mémoire
    (réservoir : R² cumulée à rappeler u(t-d), d=1..20)

Usage : python bench_sigil.py
"""
import ctypes
import time
from pathlib import Path

import numpy as np

_DLL = Path(__file__).parent / "native" / "sigil_circuit.dll"

PATTERNS = ["seal", "pentagram", "ring", "wheel", "grid", "ziggurat", "random"]

_int_p = ctypes.POINTER(ctypes.c_int32)
_float_p = ctypes.POINTER(ctypes.c_float)


class SigilCircuit:
    """Bindings ctypes : génération de topologie + pas LIF dense en C."""

    def __init__(self):
        if not _DLL.exists():
            raise FileNotFoundError(
                f"{_DLL} introuvable — compiler : gcc -O3 -march=native "
                f"-shared -o sigil_circuit.dll sigil_circuit.c")
        self.dll = ctypes.CDLL(str(_DLL))
        self.dll.sigil_edges.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            _int_p, _int_p, ctypes.c_int]
        self.dll.sigil_edges.restype = ctypes.c_int
        self.dll.sigil_step.argtypes = [
            _float_p, ctypes.c_int, _float_p, _float_p, _float_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, _float_p]
        self.dll.sigil_step.restype = None

    def edges(self, pattern: int, n: int, seed: int = 1234):
        cap = 8 * n
        src = np.zeros(cap, dtype=np.int32)
        dst = np.zeros(cap, dtype=np.int32)
        m = self.dll.sigil_edges(
            pattern, n, seed,
            src.ctypes.data_as(_int_p), dst.ctypes.data_as(_int_p), cap)
        if m < 0:
            raise ValueError(f"pattern {pattern} invalide pour n={n}")
        return src[:m].copy(), dst[:m].copy()

    def step(self, W, spikes_in, inp, V, V_th=1.0, V_reset=0.0,
             decay=0.05, out=None):
        n = W.shape[0]
        if out is None:
            out = np.zeros(n, dtype=np.float32)
        self.dll.sigil_step(
            np.ascontiguousarray(W, dtype=np.float32).ctypes.data_as(_float_p),
            n,
            np.ascontiguousarray(spikes_in, dtype=np.float32).ctypes.data_as(_float_p),
            np.ascontiguousarray(inp, dtype=np.float32).ctypes.data_as(_float_p),
            np.ascontiguousarray(V, dtype=np.float32).ctypes.data_as(_float_p),
            ctypes.c_float(V_th), ctypes.c_float(V_reset), ctypes.c_float(decay),
            out.ctypes.data_as(_float_p))
        return out


def build_W(circ, pattern, n, seed=7, rho_target=0.95, w=0.5, inh_frac=0.15):
    """W dense (n,n) depuis l'edge-list : +w, 15 % inhibiteurs, rayon 0.9."""
    rng = np.random.default_rng(seed)
    src, dst = circ.edges(pattern, n)
    W = np.zeros((n, n), dtype=np.float32)
    W[dst, src] = w
    inh = rng.random(len(src)) < inh_frac
    W[dst[inh], src[inh]] = -w
    rho = max(abs(np.linalg.eigvals(W)))
    if rho > 1e-9:
        W *= rho_target / rho
    return W, len(src)


def run_and_measure(circ, W, T=3000, seed=0, V_th=1.0, decay=0.1):
    """Drive binaire identique ; retourne (taux, burstiness, freq_dom, MC)."""
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    u = (rng.random(T) < 0.5).astype(np.float32)
    Win = rng.choice([-1.0, 1.0], size=n).astype(np.float32)  # projection entrée
    V = np.zeros(n, dtype=np.float32)
    sp = np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    inp = np.zeros(n, dtype=np.float32)
    states = np.zeros((T, n), dtype=np.float32)
    pop = np.zeros(T, dtype=np.float32)
    for t in range(T):
        # ~moitié du réseau poussée au-dessus du seuil, l'autre tenue :
        # le réservoir vit, et la topologie sculpte la dynamique.
        inp[:] = (0.5 + 1.0 * Win * u[t]).astype(np.float32)
        circ.step(W, sp, inp, V, V_th=V_th, decay=decay, out=out)
        sp, out = out, sp  # ping-pong, zéro alloc
        states[t] = sp
        pop[t] = sp.sum()
    rate = float(pop.mean() / n)
    burst = float(pop.var() / max(pop.mean(), 1e-9))
    spec = np.abs(np.fft.rfft(pop - pop.mean()))
    freqs = np.fft.rfftfreq(T, d=1.0)
    dom = float(freqs[1 + int(np.argmax(spec[1:]))]) if spec[1:].max() > 0 else 0.0
    # Capacité mémoire : rappeler u(t-d) depuis l'état (washout 200)
    S, uu = states[200:], u[200:]
    MC = 0.0
    for d in range(1, 21):
        y = uu[:-d]
        X = np.column_stack([S[d:], np.ones(len(S) - d)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss = float(((y - pred) ** 2).sum())
        tot = float(((y - y.mean()) ** 2).sum())
        MC += max(0.0, 1.0 - ss / max(tot, 1e-9))
    return rate, burst, dom, MC


def bench_speed(circ, W, steps=20000, reps=3):
    n = W.shape[0]
    V = np.zeros(n, dtype=np.float32)
    sp = np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    inp = np.zeros(n, dtype=np.float32)
    circ.step(W, sp, inp, V, out=out)  # chauffe
    best = 1e9
    for _ in range(reps):  # meilleur des reps (CPU partagé avec le fine-tune)
        t0 = time.perf_counter()
        for _ in range(steps):
            circ.step(W, sp, inp, V, out=out)
            sp, out = out, sp
        best = min(best, time.perf_counter() - t0)
    return steps / best  # pas/s


def main():
    circ = SigilCircuit()
    print(f"{'pattern':<10} {'edges':>6} {'kpas/s':>8} {'rate':>7} "
          f"{'burst':>7} {'domHz':>7} {'MC':>6}")
    for p, name in enumerate(PATTERNS):
        n = 64
        W, m = build_W(circ, p, n)
        sps = bench_speed(circ, W) / 1000.0
        rate, burst, dom, MC = run_and_measure(circ, W)
        print(f"{name:<10} {m:>6} {sps:>8.1f} {rate:>7.3f} "
              f"{burst:>7.2f} {dom:>7.4f} {MC:>6.2f}")
    print("\n--- vitesse vs taille (pattern seal) ---")
    for n in (32, 64, 128):
        W, _ = build_W(circ, 0, n)
        print(f"  n={n:<4} {bench_speed(circ, W) / 1000.0:>8.1f} kpas/s")


if __name__ == "__main__":
    main()