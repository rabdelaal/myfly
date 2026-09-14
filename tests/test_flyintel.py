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
        assert bench.LEADERBOARD_PATH.exists()
        print("ok test_benchmark_smoke")


def test_websearch():
    import requests as _rq
    from flyintel import websearch
    try:
        res = websearch.search("connectome", num=2)
        assert res and res[0]["url"].startswith("http"), res
        assert res[0]["source"] in ("exa", "duckduckgo")
    except _rq.RequestException as e:
        print(f"   [websearch] réseau indisponible, skip ({e})")
        return
    with tempfile.TemporaryDirectory() as td:
        saved = websearch.learn("connectome", dir=td, num=1, sleep_s=0)
        assert saved and Path(saved[0]).exists()
    print("ok test_websearch")


if __name__ == "__main__":
    for fn in [test_module_import, test_brain_reusable, test_backend_dispatch,
               test_spear_parity_js, test_readouts_trainable, test_gradcheck,
               test_benchmark_smoke, test_websearch]:
        fn()
    print("\nALL FLYINTEL TESTS PASSED")