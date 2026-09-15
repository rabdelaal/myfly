"""
Kit réservoir plug-and-play : les 34 motifs sigillaires comme réservoirs
prêts à l'emploi pour séries temporelles légères (style sklearn).

Usage :
    from reservoir_kit import SigilReservoir
    r = SigilReservoir(pattern="flower", n=64)
    r.fit(u_train, y_train)      # u : drive (T,), y : cible (T,)
    pred = r.predict(u_test)     # (T,)

Pas d'entraînement profond : un seul lstsq (secondes). Motifs conseillés :
mémoire -> flower/grid/iching ; horloges -> ring/pentacle.
"""
import numpy as np

from bench_sigil import SigilCircuit, build_W, PATTERNS


class SigilReservoir:
    def __init__(self, pattern: str | int = "flower", n: int = 64, seed: int = 7,
                 washout: int = 100, V_th: float = 1.0, decay: float = 0.1,
                 ridge: float = 1e-6):
        self.circ = SigilCircuit()
        pid = PATTERNS.index(pattern) if isinstance(pattern, str) else int(pattern)
        nn = self.circ.native_n(pid)
        self.n = nn if nn else n
        self.W, self.m = build_W(self.circ, pid, self.n, seed=seed)
        self.washout = washout
        self.V_th, self.decay, self.ridge = V_th, decay, ridge
        rng = np.random.default_rng(seed)
        self.Win = rng.choice([-1.0, 1.0], size=self.n).astype(np.float32)
        self.coef = None

    def _states(self, u):
        u = np.asarray(u, dtype=np.float32).ravel()
        T = len(u)
        V = np.zeros(self.n, dtype=np.float32)
        sp = np.zeros(self.n, dtype=np.float32)
        out = np.zeros(self.n, dtype=np.float32)
        inp = np.zeros(self.n, dtype=np.float32)
        S = np.zeros((T, self.n), dtype=np.float32)
        for t in range(T):
            inp[:] = (0.5 + 1.0 * self.Win * u[t]).astype(np.float32)
            self.circ.step(self.W, sp, inp, V, V_th=self.V_th,
                           decay=self.decay, out=out)
            sp, out = out, sp
            S[t] = sp  # spikes : robuste en classification de motifs
        return S

    def fit(self, u, y):
        S = self._states(u)[self.washout:]
        y = np.asarray(y, dtype=np.float32).ravel()[self.washout:]
        X = np.column_stack([S, np.ones(len(S))])
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.coef = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, u):
        assert self.coef is not None, "fit() d'abord"
        S = self._states(u)[self.washout:]
        return np.column_stack([S, np.ones(len(S))]) @ self.coef


def r2(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    n = min(len(y), len(p))
    y, p = y[-n:], p[-n:]  # aligne sur la fin (predict() enlève le washout)
    return float(1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))


if __name__ == "__main__":
    # Self-check plomberie (pas de claim de perf : le point de fonctionnement
    # input-dominé demande un vrai réglage — voir bench_sigil MC in-sample).
    rng = np.random.default_rng(0)
    T = 600
    u = (rng.random(T) < 0.5).astype(np.float32) * 2 - 1
    y = np.concatenate([np.zeros(2), u[:-2]])
    for pat in ("flower", "grid", "random"):
        r = SigilReservoir(pattern=pat, n=32, washout=20).fit(u[:500], y[:500])
        p1 = r.predict(u[500:])
        p2 = SigilReservoir(pattern=pat, n=32, washout=20).fit(u[:500], y[:500]).predict(u[500:])
        assert p1.shape == p2.shape and np.all(np.isfinite(p1))
        assert float((p1 - p2).max()) == 0.0  # déterminisme
        print(f"{pat:<8} pred_shape={p1.shape} deterministe OK")
    print("OK")