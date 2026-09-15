"""
CPG sigillaires : les motifs oscillants comme générateurs de patterns
centraux (allures). Ici en simulation (pas de hardware) : on fait tourner
un motif, on lit la phase de 6 nœuds-pattes, on imprime l'allure.

Résultat typique : le ring donne une onde voyageuse (allure métachrone,
façon mille-pattes) — mesuré, pas postulé.
"""
import numpy as np

from bench_sigil import SigilCircuit, build_W


def gait(pattern: int = 2, n: int = 64, legs: int = 6, T: int = 2000,
         seed: int = 0):
    """Retourne (phases[legs] en radians, freq_dom) des nœuds-pattes.
    Drive CONSTANT (pas de forçage temporel) : le rythme est intrinsèque
    au circuit, pas recopié de l'entrée."""
    circ = SigilCircuit()
    W, _ = build_W(circ, pattern, n, seed=seed)
    rng = np.random.default_rng(seed)
    V = np.zeros(n, dtype=np.float32)
    sp = np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    base = (1.2 + 0.05 * rng.standard_normal(n)).astype(np.float32)
    idx = [(i * n) // legs for i in range(legs)]
    S = np.zeros((T, legs), dtype=np.float32)
    for t in range(T):
        circ.step(W, sp, base, V, out=out)
        sp, out = out, sp
        S[t] = sp[idx]
    pop = S.sum(axis=1)
    spec = np.abs(np.fft.rfft(pop - pop.mean()))
    freqs = np.fft.rfftfreq(T, d=1.0)
    k = 1 + int(np.argmax(spec[1:]))
    dom = float(freqs[k])
    F = np.fft.rfft(S, axis=0)
    phases = np.angle(F[k if k < len(F) else 1])
    return np.mod(phases, 2 * np.pi), dom


def ascii_steps(phases, cycles: int = 2, width: int = 48):
    """Diagramme : une ligne par patte, # = phase d'appui."""
    rows = []
    for p in phases:
        row = "".join("#" if ((c / width * 2 * np.pi - p) % (2 * np.pi)) < np.pi
                      else "." for c in range(width * cycles))
        rows.append(row)
    return rows


if __name__ == "__main__":
    ph, dom = gait(pattern=2)  # ring
    print(f"ring: domHz={dom:.4f}")
    for i, (p, row) in enumerate(zip(ph, ascii_steps(ph))):
        print(f"  patte {i}: phase {p:5.2f} {row[:48]}")
    d = np.diff(np.sort(ph))
    spread = float(np.sort(ph)[-1] - np.sort(ph)[0])
    assert len(ph) == 6 and 0.0 < dom < 0.5
    assert spread > 1.0, spread  # phases étalées : allure émergente, pas sync
    print("OK (onde de phase mesurée sur 6 pattes)")