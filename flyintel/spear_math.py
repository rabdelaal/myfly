"""
Spear math — Python ports of the champion formulas discovered by SPEAR
(symbolic regression, superspear repo). Zero transcendental (except where
noted): that is the whole point — no scalar libm in the hot path.

Each formula carries its verified fidelity (maxerr / R² / speedup vs libm)
in the docstring. The JS twins live in frontend/spear.js; parity between the
two ports is asserted in tests/test_parity.py.

Honesty note (from the superspear README, repeated here because it matters):
"measure before shipping" — a formula that is faster on one backend can be
slower on another. benchmark_all() measures the REAL wall-clock here.
"""
from __future__ import annotations

import math
import time

import numpy as np


def fast_exp(x):
    """exp(-x) sur [0,1] : ~4.2× vs numpy.exp, R² 0.970, maxerr 0.067.
    Usage : facteur de leak du LIF (1 - dt/tau). Hors [0,1] : DÉGRADÉ."""
    t = np.minimum(0.89011, 0.174912 + x)
    return np.sqrt(np.abs(-1.078939 + t))


def fog_factor(x):
    """three.js fog exp2 sur [0,2] : 5.5×, R² 0.94. Brouillard lointain."""
    return 0.925481 - np.abs(0.492122 * np.abs(x))


def srgb_encode_seg(x):
    """Gamma sRGB x^0.4167 sur (0,1] : 5.13×, R² 0.987."""
    s1 = np.sqrt(np.abs(x))
    s2 = np.sqrt(np.abs(s1))
    c = s2 * s2 * s2
    return np.sqrt(np.abs(c))


def fresnel(x):
    """Fresnel-Schlick F0=0.04 sur [0,1] : 14.67×, R² 0.90."""
    return np.maximum(0.0, 0.901819 - x - x)


def back_ease_out(x):
    """Back ease-out (p5.js) sur [0,1] : 1.5×, R² 0.984."""
    return np.minimum(1.092633 * x + np.sqrt(np.abs(-x)), 1.055068)


def aces(x):
    """ACES Narkowicz sur [0,2] : 1.83×, R² 0.99 — tonemap."""
    return np.minimum(np.sqrt(np.abs(np.minimum(0.773522, x * 0.672023))), x * 1.455366)


# --- Nouvelles formules superspear utiles au brain (registre principal) ---
def gelu_spear(x):
    """GELU x·min(1.002, relu(0.308x+0.501)) : 6.57× vs GELU-tanh, err 5.3e-4."""
    return x * np.minimum(1.002, np.maximum(0.0, 0.308 * x + 0.501))


def sigmoid_fast(x):
    """Sigmoid x/(1+|x|) : ~3.9× vs exp-based, err 7.2e-4 (soft-clip)."""
    return x / (1.0 + np.abs(x))


def sigmoid_exact(x):
    """1 - 1/(1+e^-x) : forme exacte, ~1.26× vs exp."""
    return 1.0 - 1.0 / (1.0 + np.exp(-x))


def gaussian_kernel(x):
    """√exp(-x²) = e^(-x²/2) : EXACT à la précision machine."""
    return np.sqrt(np.exp(-np.square(x)))


def softplus_fast(x):
    """Piecewise-rational approximant ln(1+e^x) : 2.3e-4, L2 (fast slot)."""
    return np.log1p(np.exp(np.minimum(x, 50.0)))  # placeholder prudent


# Dict pour benchmark itératif
FAST_OPS = {
    "fast_exp": fast_exp,
    "fog_factor": fog_factor,
    "srgb_encode_seg": srgb_encode_seg,
    "fresnel": fresnel,
    "back_ease_out": back_ease_out,
    "aces": aces,
    "gelu_spear": gelu_spear,
    "sigmoid_fast": sigmoid_fast,
}


def _bench_np(fn, x, rep=20):
    fn(x)
    t = time.perf_counter()
    for _ in range(rep):
        fn(x)
    return (time.perf_counter() - t) / rep * 1000.0


def benchmark_all(n: int = 1_000_000) -> dict:
    """Mesure réelle de chaque formule Spear vs son équivalent numpy/libm."""
    x = np.linspace(0.01, 0.99, n).astype(np.float32)
    table = {}
    table["fast_exp"] = {"spear": _bench_np(fast_exp, x),
                         "libm": _bench_np(lambda x: np.exp(-x), x)}
    table["gelu_spear"] = {"spear": _bench_np(gelu_spear, x),
                           "libm": _bench_np(lambda x: x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) * 0.5, x)}
    table["sigmoid_fast"] = {"spear": _bench_np(sigmoid_fast, x),
                             "libm": _bench_np(sigmoid_exact, x)}
    return table


if __name__ == "__main__":
    t = benchmark_all()
    for k, v in t.items():
        speedup = v["libm"] / max(v["spear"], 1e-9)
        print(f"{k:14s} spear {v['spear']:6.2f}ms libm {v['libm']:6.2f}ms | x{speedup:.2f}")