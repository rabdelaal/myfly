"""
Accelerated backends for the fly's readout networks.

Dispatch between torch (MKL) and SpearVM (spur-math, AVX2) on the REAL
measured wall-clock of THIS machine — never on the README numbers. Both
spear-kernels and SpearVM are honest that transcendental costs differ per
backend, so we calibrate once at import and expose:

  - gelu_fast(x)   : best of torch / spear, element-wise
  - tanh_fast(x)   : best of torch / spear
  - matmul_fast    : torch always wins on matmul here (MKL), kept as a shim
  - benchmark_backends() : returns the measured table (used by the leaderboard)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .spear_math import FAST_OPS  # noqa: E402

_SPEAR = None  # lazy spur_math import
_HAS_SPEAR = False


def _load_spear():
    """Tente d'importer spur_math; échoue proprement si absent/pas AVX2."""
    global _SPEAR, _HAS_SPEAR
    if _HAS_SPEAR or _SPEAR is not None:
        return _SPEAR
    try:
        import spur_math  # type: ignore

        _SPEAR = spur_math
        _HAS_SPEAR = True
    except Exception:
        _SPEAR = None
        _HAS_SPEAR = False
    return _SPEAR


def _bench(fn, rep: int = 30) -> float:
    """ms par appel, mesuré sur un tenseur fixe."""
    fn()
    t = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t) / rep * 1000.0


_MEASURED: dict | None = None


def benchmark_backends(n: int = 1_000_000, rep: int = 30) -> dict:
    """Mesure réelle torch vs SpearVM sur ce CPU. Table utilisée par le
    leaderboard et pour choisir le dispatch par défaut."""
    global _MEASURED
    torch.set_num_threads(os.cpu_count() or 4)
    x = np.random.randn(n).astype(np.float32)
    xt = torch.from_numpy(x)

    table = {"n": n, "device": "cpu", "torch_threads": torch.get_num_threads()}
    table["gelu_torch_ms"] = _bench(lambda: torch.nn.functional.gelu(xt), rep)
    table["tanh_torch_ms"] = _bench(lambda: torch.tanh(xt), rep)

    sm = _load_spear()
    if sm is not None:
        table["gelu_spear_ms"] = _bench(lambda: sm.gelu(x), rep)
        table["tanh_spear_ms"] = _bench(lambda: sm.tanh(x), rep)
        table["spear_available"] = True
    else:
        table["gelu_spear_ms"] = None
        table["tanh_spear_ms"] = None
        table["spear_available"] = False

    # dispatch par défaut : le plus rapide pour chaque op
    table["use_spear_gelu"] = bool(
        table.get("spear_available")
        and table["gelu_spear_ms"] < table["gelu_torch_ms"]
    )
    table["use_spear_tanh"] = bool(
        table.get("spear_available")
        and table["tanh_spear_ms"] < table["tanh_torch_ms"]
    )

    # Formules Spear algébriques (registre superspear) vs leurs références
    from .spear_math import _bench_np, gelu_spear, sigmoid_fast

    def _gelu_libm(x):
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    xg = np.random.randn(n).astype(np.float32)
    table["gelu_spear_ms"] = _bench_np(lambda x: gelu_spear(x).astype(np.float32), xg, rep)
    table["gelu_libm_ms"] = _bench_np(_gelu_libm, xg, rep)
    table["use_spear_gelu_algebraic"] = bool(
        table["gelu_spear_ms"] < table["gelu_libm_ms"])
    xs = np.random.rand(n).astype(np.float32)
    from .spear_math import sigmoid_fast, sigmoid_exact
    table["sigmoid_spear_ms"] = _bench_np(lambda x: sigmoid_fast(x).astype(np.float32), xs, rep)
    table["sigmoid_libm_ms"] = _bench_np(lambda x: sigmoid_exact(x).astype(np.float32), xs, rep)
    table["use_spear_sigmoid"] = bool(table["sigmoid_spear_ms"] < table["sigmoid_libm_ms"])

    _MEASURED = table
    return table


def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    """GELU élément-par-élément, dispatch torch/spear selon la mesure."""
    if _MEASURED is None:
        benchmark_backends()
    if _MEASURED.get("use_spear_gelu") and x.device.type == "cpu":
        sm = _load_spear()
        if sm is not None:
            return torch.from_numpy(sm.gelu(x.detach().cpu().numpy().astype(np.float32)))
    return torch.nn.functional.gelu(x)


def tanh_fast(x: torch.Tensor) -> torch.Tensor:
    """tanh élément-par-élément, dispatch torch/spear selon la mesure."""
    if _MEASURED is None:
        benchmark_backends()
    if _MEASURED.get("use_spear_tanh") and x.device.type == "cpu":
        sm = _load_spear()
        if sm is not None:
            return torch.from_numpy(sm.tanh(x.detach().cpu().numpy().astype(np.float32)))
    return torch.tanh(x)


def matmul_fast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """matmul : torch/MKL toujours retenu ici (le tuilé SpearVM perd sur les
    vieux CPU 2 cœurs vs MKL — mesuré ×0.10). Gardé en shim pour un futur
    dispatch GPU/NPU."""
    return a @ b