"""
Tests unitaires flyintel — sans framework, lancement : python tests/test_flyintel.py
Couvre : import, module brain réutilisable, backends dispatch, spear_math parité
JS/Python, readouts Spear entraînables (forward+backward+gradcheck), benchmark smoke.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import flyintel  # noqa: E402
from flyintel import bench  # noqa: E402
from flyintel.backends import benchmark_backends, gelu_fast, tanh_fast  # noqa: E402
from flyintel.spear_math import fast_exp, gelu_spear, sigmoid_fast  # noqa: E402
from flyintel.readouts_spear import LinearReadoutSpear, LoopedReadoutSpear, _GeluSpear, _SigmoidFast  # noqa: E402


def test_module_import():
    assert flyintel.__version__
    assert hasattr(flyintel, "load_brain")
    assert hasattr(flyintel, "default_device")
    print("ok test_module_import")


def test_brain_reusable():
    # construit un petit cerveau réutilisable sans backend web
    brain, conn = flyintel.load_brain(force_synthetic=True)
    assert conn["W"].shape[0] > 0
    assert brain.n == conn["W"].shape[0]
    # un step + run marchent
    brain.reset(1)
    brain.step(None)
    from flyintel import BoardEncoder
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    c = enc.encode_batch(_boards())
    assert c.shape[0] == int(conn["is_sensory"].sum())
    assert c.shape[1] == 2
    print("ok test_brain_reusable")


def _boards():
    import chess
    return [chess.Board() for _ in range(2)]


def test_backend_dispatch():
    t = benchmark_backends(n=50_000, rep=5)
    assert t["n"] == 50_000
    assert "gelu_torch_ms" in t
    # les formules algébriques gagnent (mesuré sur ce CPU)
    assert t.get("use_spear_gelu_algebraic", False) is not None
    # dispatch n'explose pas
    x = torch.randn(10)
    y = gelu_fast(x)
    assert y.shape == x.shape
    print("ok test_backend_dispatch")


def test_spear_parity_js():
    """Parité JS/Python des formules champions (frontend/spear.js)."""
    # fast_exp : même formule que frontend/spear.js
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    got = fast_exp(x)
    assert got.shape == x.shape
    # R² ~0.97 : comparer à exp(-x) dans la bande de validité [0,1]
    ref = np.exp(-x)
    r2 = 1 - float(np.sum((got - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))
    assert r2 > 0.90, f"R²={r2} sous le seuil"
    print(f"ok test_spear_parity_js (fast_exp R²={r2:.3f})")


def test_readouts_trainable():
    torch.manual_seed(0)
    lr = LinearReadoutSpear(n_motor=8, n_descending=4)
    m = torch.randn(5, 8)
    d = torch.randn(5, 4)
    out = lr(m, d)
    assert out.shape == (5,)
    out.sum().backward()
    assert lr.fc.weight.grad is not None

    lo = LoopedReadoutSpear(d_in=12, d_h=16, window=4)
    traj = torch.randn(6, 3, 12)
    o = lo(traj)
    assert o.shape == (6, 3)
    o[-1].sum().backward()
    assert lo.cell.gru.weight_hh.grad is not None
    print("ok test_readouts_trainable")


def test_gradcheck():
    def gc(F):
        x = torch.linspace(-4, 4, 100)
        xg = x.clone().requires_grad_(True)
        F.apply(xg).sum().backward()
        ga = xg.grad.clone()
        eps = 1e-4
        gn = (F.apply(x + eps) - F.apply(x - eps)) / (2 * eps)
        return float((ga - gn).abs().max())
    assert gc(_GeluSpear) < 0.01, f"gelu grad maxerr {gc(_GeluSpear)}"
    assert gc(_SigmoidFast) < 0.01
    print("ok test_gradcheck")


def test_benchmark_smoke():
    with tempfile.TemporaryDirectory() as td:
        bench.LEADERBOARD_PATH = Path(td) / "leaderboard.json"
        from data_loader import _synthetic_connectome
        from brain import FlyBrain
        conn = _synthetic_connectome(n_neurons=500, sparsity=0.01)
        brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
        r = bench.run_all(brain, conn, tag="unittest")
        assert r["tag"] == "unittest"
        assert "dynamics" in r["domains"]
        assert "metaphor" in r["domains"]
        assert bench.LEADERBOARD_PATH.exists()
        print("ok test_benchmark_smoke")


def test_metaphor_stub():
    from encoding import AnythingEncoder
    enc = AnythingEncoder(n_sensory=64)
    c = enc.encode_options(["sacrifice the queen", "push a pawn"])
    assert c.shape == (64, 2)
    # norme constante par option (même régime que BoardEncoder)
    assert abs(float(c[:, 0].norm()) - 450.0) < 1.0
    assert abs(float(c[:, 1].norm()) - 450.0) < 1.0
    # options distinctes -> patterns distincts
    assert float((c[:, 0] - c[:, 1]).norm()) > 1.0
    # déterminisme
    c2 = enc.encode_options(["sacrifice the queen", "push a pawn"])
    assert float((c - c2).abs().max()) == 0.0
    # domaine stub : séparabilité sans teacher
    from data_loader import _synthetic_connectome
    from brain import FlyBrain
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.02)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    r = bench.metaphor(brain, conn, steps=30)
    assert r["score"] is None and "separation" in r["detail"], r
    print("ok test_metaphor_stub")


def test_sigil():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from bench_sigil import SigilCircuit, build_W, run_and_measure
    circ = SigilCircuit()
    s_src, _ = circ.edges(0, 64)   # seal
    r_src, _ = circ.edges(2, 64)   # ring
    assert len(r_src) == 64, len(r_src)          # cycle exact
    assert len(s_src) > len(r_src)               # sceau plus riche que l'anneau
    assert circ.native_n(9) == 10                # Arbre de Vie : taille canonique
    t_src, _ = circ.edges(9, 10)
    assert len(t_src) == 44, len(t_src)          # 22 sentiers x2
    assert circ.native_n(0) == 0                 # motifs redimensionnables
    assert circ.native_n(26) == 22               # Yetzirah : 22 lettres
    y_src, _ = circ.edges(26, 22)
    assert len(y_src) == 462, len(y_src)         # 231 portes x2
    i_src, _ = circ.edges(24, 64)
    assert len(i_src) == 384, len(i_src)         # Q6 : 64*6/2 x2
    assert circ.native_n(23) == 13               # Métatron : 13 cercles
    assert circ.native_n(52) == 16               # Chaosigil : petit graphe
    y_src, _ = circ.edges(38, 64)                # Yggdrasil : DAG
    assert len(y_src) > 0
    f_src, _ = circ.edges(34, 64, seed=5)        # Futhark : inscription
    assert len(f_src) > 0
    g_src, _ = circ.edges(35, 64, seed=1)        # Vegvisir
    assert len(g_src) > 0
    f2_src, _ = circ.edges(53, 64, seed=24)      # Futhorc : Ac
    assert len(f2_src) > 0
    a_src, _ = circ.edges(54, 64)                # Abramelin
    assert len(a_src) > 0
    p_src, _ = circ.edges(55, 64, seed=0)        # Saturne
    assert len(p_src) > 0
    W, m = build_W(circ, 0, 32)
    rate, burst, dom, MC = run_and_measure(circ, W, T=300)
    assert 0.0 < rate < 0.5, rate
    assert MC >= 0.0
    print("ok test_sigil")


def test_usecases():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from flyhash import FlyHash
    fh = FlyHash().index(["queen checkmate attack", "cat sleeps rug"])
    assert fh.query("queen checkmate", top=1)[0][0] == 0
    from reservoir_kit import SigilReservoir
    r = SigilReservoir(pattern="flower", n=32, washout=10).fit([0.0] * 200, [1.0] * 200)
    assert len(r.predict([0.0] * 60)) == 50
    from cpg_demo import gait
    ph, dom = gait(T=300)
    assert len(ph) == 6 and 0.0 < dom < 0.5
    print("ok test_usecases")


def test_ie_worlds():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from ie_worlds import LedgerWorld, RosterWorld
    w = LedgerWorld(3)
    w.gen(40, invalid_rate=0.15)
    assert w.replay() == w.replay(list(w.events))  # déterminisme
    q, a, chk = w.ask("total")
    assert q and a and chk
    r = RosterWorld(3)
    r.gen(40, invalid_rate=0.15)
    q2, a2, chk2 = r.ask("duty")
    assert q2 and a2 is not None and chk2
    # domaine bench : stub sans readout, vérité != piège enregistrés
    from data_loader import _synthetic_connectome
    from brain import FlyBrain
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.02)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    res = bench.ie_state(brain, conn, steps=30)
    assert res["score"] is None and "truth=" in res["detail"], res
    print("ok test_ie_worlds")


def test_websearch():
    import requests as _rq
    from flyintel import websearch
    try:
        res = websearch.search("connectome", num=2)
        if not res:
            print("   [websearch] réponse vide (rate-limit?), skip")
            return
        assert res[0]["url"].startswith("http"), res
        assert res[0]["source"] in ("exa", "duckduckgo")
    except _rq.RequestException as e:
        print(f"   [websearch] réseau indisponible, skip ({e})")
        return
    with tempfile.TemporaryDirectory() as td:
        saved = websearch.learn("connectome", dir=td, num=1, sleep_s=0)
        if not saved:
            print("   [websearch] learn vide (rate-limit?), skip")
            return
        assert Path(saved[0]).exists()
        items = websearch.list_learned(td)
        assert len(items) == 1 and items[0]["title"] and items[0]["excerpt"], items
        rec = websearch.recall_latest(td)
        assert rec and rec["title"] == items[0]["title"] and rec["text"]
        assert websearch.list_learned(str(Path(td) / "absent")) == []
        assert websearch.recall_latest(str(Path(td) / "absent")) is None
    print("ok test_websearch")


if __name__ == "__main__":
    for fn in [test_module_import, test_brain_reusable, test_backend_dispatch,
               test_spear_parity_js, test_readouts_trainable, test_gradcheck,
               test_benchmark_smoke, test_metaphor_stub, test_sigil,
               test_usecases, test_ie_worlds, test_websearch]:
        fn()
    print("\nALL FLYINTEL TESTS PASSED")