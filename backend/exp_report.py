"""
Expériences rapides (secondes chacune) pour le rapport améliorations/usecases :
  E1 : sweep gain d'entrée du réservoir (levier du kit)
  E2 : sweep FlyHash (n, k) + taux de faux positifs vs théorie k²/n
  E3 : stabilité du ranking codex (MC) sur 2 seeds de construction
  E4 : preuve d'acyclicité d'yggdrasil (Kahn) + témoin cyclique (ring)
  E5 : comparatif CPG intrinsèque ring/pentacle/flower
Usage : python -B exp_report.py
"""
import numpy as np

from bench_sigil import SigilCircuit, build_W, run_and_measure, PATTERNS
from reservoir_kit import SigilReservoir, r2
from flyhash import FlyHash


def E1():
    print("== E1 : gain d'entrée Win x [0.3, 1.0, 2.0], rappel t-2 held-out ==")
    rng = np.random.default_rng(0)
    T = 800
    u = (rng.random(T) < 0.5).astype(np.float32) * 2 - 1
    y = np.concatenate([np.zeros(2), u[:-2]])
    for pat in ("flower", "random"):
        row = []
        for g in (0.3, 1.0, 2.0):
            r = SigilReservoir(pattern=pat, n=32, washout=20)
            r.Win = (g * r.Win).astype(np.float32)
            r.fit(u[:650], y[:650])
            row.append(f"g={g}:{r2(y[650:], r.predict(u[650:])):+.3f}")
        print(f"  {pat:<8} " + " ".join(row))


def E2():
    print("== E2 : FlyHash precision@1 (requêtes paraphrases inédites) + FP ==")
    docs = ["the queen sacrifices for checkmate", "queen checkmate attack",
            "the cat sleeps on the rug", "a dog naps on the carpet",
            "quantum entanglement of photons", "photon quantum physics",
            "drosophila mushroom body memory", "fly brain memory neurons"]
    # requêtes NOUVELLES (jamais indexées) -> doc attendu
    probes = [("queen gives mate", (0, 1)), ("kitten carpet nap", (2, 3)),
              ("entangled light particles", (4, 5)), ("insect brain recall", (6, 7))]
    for ns, k in ((256, 13), (512, 26), (512, 51)):
        fh = FlyHash(n_sensory=ns, k=k).index(docs)
        hits = sum(fh.query(q, top=1)[0][0] in want for q, want in probes)
        print(f"  n={ns} k={k}: precision@1 = {hits}/{len(probes)}")
    rng = np.random.default_rng(1)
    va = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing"]
    vb = ["zebra", "quasar", "xylophone", "fjord", "glyph", "quartz", "nymph"]
    fh = FlyHash(n_sensory=512, k=26)
    ov = []
    for _ in range(20):  # vocabulaires DISJOINTS : pures collisions de hash
        a = " ".join(rng.choice(va, size=6))
        b = " ".join(rng.choice(vb, size=6))
        ha, hb = fh._hash(a).astype(int), fh._hash(b).astype(int)
        ov.append(int((ha * hb).sum()))
    print(f"  faux positifs (vocabulaires disjoints): overlap moyen={np.mean(ov):.2f} "
          f"(théorie k²/n={26*26/512:.2f})")


def E3():
    print("== E3 : ranking MC (T=1500) sur seeds construction {7, 21} ==")
    circ = SigilCircuit()
    pats = ["flower", "grid", "iching", "pentagram", "random", "isa"]
    for seed in (7, 21):
        row = []
        for name in pats:
            W, _ = build_W(circ, PATTERNS.index(name), 64, seed=seed)
            _, _, _, MC = run_and_measure(circ, W, T=1500, seed=0)
            row.append(f"{name}={MC:.2f}")
        print(f"  seed {seed}: " + " ".join(row))


def E4():
    print("== E4 : acyclicité (Kahn) : yggdrasil DAG ? ring cyclique ? ==")
    from collections import deque
    circ = SigilCircuit()

    def has_cycle(pid, n):
        src, dst = circ.edges(pid, n)
        indeg = np.zeros(n, dtype=int)
        for d in dst:
            indeg[d] += 1
        q = deque([i for i in range(n) if indeg[i] == 0])
        seen = 0
        adj = [[] for _ in range(n)]
        for s, d in zip(src, dst):
            adj[s].append(d)
        while q:
            i = q.popleft()
            seen += 1
            for j in adj[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
        return seen != n  # True = cycle restant

    print(f"  yggdrasil a un cycle ? {has_cycle(38, 64)} (attendu False)")
    print(f"  ring a un cycle ? {has_cycle(2, 64)} (attendu True)")
    assert not has_cycle(38, 64) and has_cycle(2, 64)
    print("  OK")


def E5():
    print("== E5 : CPG intrinsèque (drive constant) ring/pentacle/flower ==")
    from cpg_demo import gait
    for pid, name in ((2, "ring"), (1, "pentagram"), (22, "flower")):
        ph, dom = gait(pattern=pid, T=1500)
        spread = float(ph.max() - ph.min())
        print(f"  {name:<10} domHz={dom:.4f} spread={spread:.2f} rad")


def E6():
    print("== E6 : rappel après RÉVISION (réservoir hanté par l'ancien ?) ==")
    print("   drive constant par morceaux + écrasements ; cible = valeur à t-3.")
    print("   Hypothèse : les boucles retiennent le périmé (moins bien).")
    from reservoir_kit import SigilReservoir, r2
    rng = np.random.default_rng(3)
    T, seg = 1200, 40
    u = np.repeat(rng.choice([-1.0, 1.0], size=T // seg), seg).astype(np.float32)
    d = 3
    y = np.concatenate([np.zeros(d), u[:-d]])
    for name in ("flower", "isa", "yggdrasil", "random"):
        pid = PATTERNS.index(name)
        nn = SigilCircuit().native_n(pid)
        r = SigilReservoir(pattern=name, n=nn if nn else 32, washout=20)
        r.fit(u[:950], y[:950])
        print(f"  {name:<10} R2={r2(y[950:], r.predict(u[950:])):+.3f}")


if __name__ == "__main__":
    E1()
    E2()
    E3()
    E4()
    E5()
    E6()
    print("EXPERIENCES OK")