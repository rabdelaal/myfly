"""
Bench Python du kernel HTC-Core v2.1 (GEMV ternaire AVX2) :
cross-check bit-exact vs numpy + débit batched via ctypes.

Usage projet : primitive prête pour l'inférence du readout quantifié
{-1,0,+1} quand le teacher arrivera (pas pour MaleCNS : poids non ternaires).
Usage : python bench_htc_core.py
"""
import ctypes
import time
from pathlib import Path

import numpy as np

_DLL = Path(__file__).parent / "native" / "htc_core.dll"

_u8_p = ctypes.POINTER(ctypes.c_uint8)
_u32_p = ctypes.POINTER(ctypes.c_uint32)
_i32_p = ctypes.POINTER(ctypes.c_int32)


class HTCCore:
    def __init__(self):
        if not _DLL.exists():
            raise FileNotFoundError(f"{_DLL} introuvable — compiler : "
                                    f"gcc -O3 -mavx2 -shared -o htc_core.dll htc_core.c")
        self.dll = ctypes.CDLL(str(_DLL))
        for fn in ("htc_gemv_v21_bitplane_online_4x64",):
            f = getattr(self.dll, fn)
            f.argtypes = [_u8_p, _u32_p, _u32_p, _i32_p, _i32_p]
            f.restype = None

    def gemv4x64(self, a_int8, w_tern, bias=0):
        """a (64,) int8 signé, w (4,64) ternaire -> y (4,) int32, bit-exact."""
        a = np.asarray(a_int8, dtype=np.int8)
        w = np.asarray(w_tern, dtype=np.int8)
        u = (a.astype(np.int16) ^ 0x80).astype(np.uint8)  # offset bijectif
        pp = np.zeros(8, dtype=np.uint32)
        pn = np.zeros(8, dtype=np.uint32)
        c = np.zeros(4, dtype=np.int32)
        for r in range(4):
            pos = (w[r] > 0)
            neg = (w[r] < 0)
            for i in np.nonzero(pos)[0]:
                pp[2 * r + (i // 32)] |= np.uint32(1 << (i % 32))
            for i in np.nonzero(neg)[0]:
                pn[2 * r + (i // 32)] |= np.uint32(1 << (i % 32))
            c[r] = np.int32((int(pos.sum()) - int(neg.sum())) * 128 - bias)
        y = np.zeros(4, dtype=np.int32)
        self.dll.htc_gemv_v21_bitplane_online_4x64(
            u.ctypes.data_as(_u8_p), pp.ctypes.data_as(_u32_p),
            pn.ctypes.data_as(_u32_p), c.ctypes.data_as(_i32_p),
            y.ctypes.data_as(_i32_p))
        return y


def main():
    htc = HTCCore()
    rng = np.random.default_rng(0)
    # cross-check vs numpy, cas limites -128 inclus
    bad = 0
    for t in range(300):
        if t % 3 == 0:
            a = rng.integers(-128, 128, size=64, dtype=np.int8)
        elif t % 3 == 1:
            a = np.full(64, -128, dtype=np.int8)
        else:
            a = rng.choice(np.array([-128, -1, 0, 1, 127], dtype=np.int8), size=64)
        w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(4, 64))
        bias = int(rng.integers(-1000, 1000))
        y = htc.gemv4x64(a, w, bias)
        ref = (a.astype(np.int64)[:, None] * w.astype(np.int64).T).sum(axis=0) + bias
        if not np.array_equal(y.astype(np.int64), ref):
            bad += 1
    print(f"cross-check numpy : {300 - bad}/300 bit-exact (-128 inclus)")
    assert bad == 0
    # débit batched (inclut l'overhead ctypes : borne basse honnête)
    a = rng.integers(-128, 128, size=64, dtype=np.int8)
    w = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(4, 64))
    htc.gemv4x64(a, w)
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        htc.gemv4x64(a, w)
    dt = time.perf_counter() - t0
    print(f"débit batched : {N * 4 / dt / 1e6:.2f} Mprod/s "
          f"({dt / N * 1e6:.1f} µs/appel, overhead ctypes inclus)")


if __name__ == "__main__":
    main()