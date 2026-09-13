"""
Bindings ctypes pour le kernel LIF natif (lif_kernel.dll, compilé en C).

Le kernel a été vérifié bit-exact contre une référence naïve C sur 1 000 000
de réseaux aléatoires (lif_verify.exe) ; ce module le vérifie en plus contre
la référence torch de brain.py, puis le branche comme backend optionnel.
"""
import ctypes
from pathlib import Path

import numpy as np

_DLL_PATH = Path(__file__).parent / "native" / "lif_kernel.dll"

_int_p = ctypes.POINTER(ctypes.c_int32)
_float_p = ctypes.POINTER(ctypes.c_float)


def _load_dll():
    if not _DLL_PATH.exists():
        raise FileNotFoundError(
            f"{_DLL_PATH} introuvable — compiler : gcc -O3 -march=native "
            f"-ffp-contract=off -shared -o lif_kernel.dll lif_kernel.c"
        )
    dll = ctypes.CDLL(str(_DLL_PATH))
    dll.lif_step.argtypes = [
        _int_p, _int_p, _float_p,          # indptr, indices, data
        ctypes.c_int32, ctypes.c_int32,    # n, B
        _float_p, _float_p, _float_p,      # spikes, V, isyn (work buffer)
        _float_p,                          # I_full (n,B) ou NULL
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,    # neuromod, tau_m, dt, V_th, V_reset
        _float_p,                          # spikes_out
    ]
    dll.lif_step.restype = None
    return dll


class NativeLIF:
    """Backend LIF natif : même sémantique que FlyBrain.step, zéro allocation."""

    def __init__(self):
        self.dll = _load_dll()

    def step(self, indptr, indices, data, spikes, V, isyn, I_full,
             neuromod, tau_m, dt, V_th, V_reset, spikes_out):
        n = np.int32(V.shape[0])
        B = np.int32(V.shape[1])
        self.dll.lif_step(
            indptr.ctypes.data_as(_int_p),
            indices.ctypes.data_as(_int_p),
            data.ctypes.data_as(_float_p),
            n, B,
            spikes.ctypes.data_as(_float_p),
            V.ctypes.data_as(_float_p),
            isyn.ctypes.data_as(_float_p),
            I_full.ctypes.data_as(_float_p) if I_full is not None else None,
            ctypes.c_float(neuromod), ctypes.c_float(tau_m), ctypes.c_float(dt),
            ctypes.c_float(V_th), ctypes.c_float(V_reset),
            spikes_out.ctypes.data_as(_float_p),
        )


def verify_vs_torch(n_trials: int = 300, tol: float = 1e-4) -> tuple[int, int]:
    """
    Vérification croisée contre la référence torch de brain.py sur des réseaux
    aléatoires (formule float32, ordres d'accumulation différents -> tolérance).
    Retourne (échecs, essais).
    """
    import torch
    import scipy.sparse as sp

    native = NativeLIF()
    rng = np.random.default_rng(0)
    failures = 0
    for _ in range(n_trials):
        n = int(rng.integers(4, 200))
        B = int(rng.integers(1, 9))
        density = rng.uniform(0.01, 0.5)
        M = sp.random(n, n, density=density, format="csr", dtype=np.float32,
                      random_state=int(rng.integers(1 << 30)))
        M.data[:] = (rng.uniform(-1, 1, size=M.nnz) * 2).astype(np.float32)
        neuromod = float(rng.uniform(0, 2))
        tau_m = float(rng.uniform(1, 40))
        dt = float(rng.uniform(0.1, 5))
        V_th = float(rng.uniform(0.5, 1.5))
        V_reset = float(rng.choice([0.0, -0.2]))

        spikes_np = (rng.random((n, B)) < 0.3).astype(np.float32)
        V_np = rng.uniform(0, V_th * 1.2, size=(n, B)).astype(np.float32)
        I_np = (rng.uniform(-1.5, 1.5, size=(n, B))).astype(np.float32)
        isyn = np.zeros_like(V_np)
        out_np = np.zeros_like(V_np)
        V_native = V_np.copy()

        native.step(M.indptr.astype(np.int32), M.indices.astype(np.int32),
                    np.ascontiguousarray(M.data), spikes_np, V_native, isyn,
                    I_np, neuromod, tau_m, dt, V_th, V_reset, out_np)

        # Référence torch (même formule que FlyBrain.step)
        W = torch.sparse_csr_tensor(
            torch.tensor(M.indptr), torch.tensor(M.indices),
            torch.tensor(M.data), size=M.shape)
        sp_t = torch.tensor(spikes_np)
        I_syn = neuromod * torch.sparse.mm(W, sp_t) + torch.tensor(I_np)
        V_t = torch.tensor(V_np) + (-torch.tensor(V_np) + I_syn) * (dt / tau_m)
        spiked_t = V_t >= V_th
        V_ref = torch.where(spiked_t, torch.full_like(V_t, V_reset), V_t)
        out_ref = spiked_t.float()

        dv = float((torch.tensor(V_native) - V_ref).abs().max())
        ds = float((torch.tensor(out_np) - out_ref).abs().max())
        if dv > tol or ds > tol:
            failures += 1

    return failures, n_trials
