"""
Opérateur Collatz-Fibonacci : T(n) = n/2 si pair, (3n + F(n mod 10))//2 sinon,
avec F = Fibonacci(0..9) = [0,1,1,2,3,5,8,13,21,34].

Reproduction computationally de l'expérience : découverte NON biaisée des
cycles (détection générale, on ne suppose PAS qu'il n'y en a que deux),
stats de bassins, vérification des Théorèmes 1 (isomorphisme 5Z ~ Collatz),
2 (résidus 3,9 aspirés en 1 étape), et du rayon spectral de la matrice Q.

Usage : python -B collatz_fib.py [N]   (défaut N=50000, ~1 min)
"""
import sys

import numpy as np

F = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)


def T(n: int) -> int:
    if n & 1:
        return (3 * n + F[n % 10]) // 2
    return n // 2


def collatz(n: int) -> int:
    return n // 2 if n % 2 == 0 else (3 * n + 1) // 2


def scan(N: int, cap: int = 200000):
    """Bassin de chaque n in [1..N] + transitions de chiffres pour Q.
    Retourne (basins, cycles, transitions, max_steps, nonterm)."""
    cache: dict = {}
    trans = np.zeros((10, 10), dtype=np.int64)
    max_steps = 0
    nonterm = []
    Tloc = T
    for start in range(1, N + 1):
        n = start
        path = []
        pos = {}
        while n not in cache:
            if n in pos:  # cycle découvert (sans a priori)
                cyc = path[pos[n]:]
                cyc = tuple(sorted(set(cyc)))
                for x in path:
                    cache[x] = cyc
                break
            pos[n] = len(path)
            path.append(n)
            prev, n = n, Tloc(n)
            trans[prev % 10, n % 10] += 1
            if len(path) > cap:
                nonterm.append(start)
                for x in path:
                    cache[x] = None
                break
        if n in cache and cache[n] is not None:
            b = cache[n]
            for x in path:
                cache[x] = b
        max_steps = max(max_steps, len(path))
    basins = [cache[i] for i in range(1, N + 1)]
    cycles = sorted(set(basins))
    return basins, cycles, trans, max_steps, nonterm


def theorem1(limit: int = 5000):
    """T(5m)/5 == collatz(m) pour tout m (5Z invariant + isomorphe)."""
    for m in range(1, limit + 1):
        assert T(5 * m) % 5 == 0, m
        assert T(5 * m) // 5 == collatz(m), m
    return limit


def theorem2(limit: int = 20000):
    """10k+3 et 10k+9 tombent dans 5Z en exactement 1 étape."""
    for k in range(limit):
        assert T(10 * k + 3) % 5 == 0, k
        assert T(10 * k + 9) % 5 == 0, k
    return 2 * limit


S_DIGITS = (1, 2, 4, 6, 7, 8)


def spectral(trans) -> tuple:
    """Q empirique sur S + rayon spectral."""
    idx = {d: i for i, d in enumerate(S_DIGITS)}
    Q = np.zeros((6, 6))
    for a in S_DIGITS:
        row = trans[a][list(S_DIGITS)].astype(float)
        s = trans[a].sum()
        if s > 0:
            Q[idx[a]] = row / s  # sous-stochastique (reste = absorption 5Z)
    lam = max(abs(np.linalg.eigvals(Q)))
    return Q, float(lam)


def main(N: int):
    basins, cycles, trans, ms, nonterm = scan(N)
    print(f"nombres : 1..{N} | cycles découverts : {cycles}")
    print(f"non-terminés (cap) : {len(nonterm)} | pas max : {ms}")
    assert not nonterm, nonterm[:5]
    b510 = {(5, 10), (10, 5)}
    c1 = sum(1 for b in basins[:2000] if set(b) == {5, 10})
    c2 = sum(1 for b in basins[2000:] if set(b) == {5, 10})
    print(f"bassin (5,10) sur [1..2000] : {c1 / 2000 * 100:.1f}% "
          f"(papier : 94,8%)")
    if N > 2000:
        print(f"bassin (5,10) sur [2001..{N}] : {c2 / (N - 2000) * 100:.1f}% "
              f"(papier : 99,0%)")
    print(f"Théorème 1 : isomorphisme vérifié sur {theorem1()} valeurs de m")
    print(f"Théorème 2 : {theorem2()} résidus 3/9 aspirés en 1 étape, 0 échec")
    Q, lam = spectral(trans)
    print(f"rayon spectral Q empirique : {lam:.4f} (papier : 0.8774)")
    print("Q empirique (lignes 1,2,4,6,7,8) :")
    for row in Q:
        print("   " + " ".join(f"{x:.2f}" for x in row))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50000)
