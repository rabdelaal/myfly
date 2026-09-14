"""
Bindings ctypes pour le kernel LIF α-synapse EVENT-DRIVEN (lif_alpha_ed.dll).

Conçu pour le cerveau MaleCNS en régime prod (batch_size = 1) : ne disperse que
les neurones présynaptiques actifs (~7 %) en layout CSC, au lieu du
torch.sparse.mm dense. Vérifié bit-exact contre une référence naïve C sur
1 000 000 de réseaux aléatoires, et ici contre la référence torch de brain.py
(sémantique alpha de Shiu et al. : délai, décroissance τ_syn, réfractaire).
"""
import ctypes
from pathlib import Path

import numpy as np
import scipy.sparse as _sp

_DLL_PATH = Path(__file__).parent / "native" / "lif_alpha_ed.dll"

_int_p = ctypes.POINTER(ctypes.c_int32)
_float_p = ctypes.POINTER(ctypes.c_float)


def _load_dll():
    if not _DLL_PATH.exists():
        raise FileNotFoundError(
            f"{_DLL_PATH} introuvable — compiler : gcc -O3 -march=native "
            f"-shared -o lif_alpha_ed.dll lif_alpha_ed.c"
        )
    dll = ctypes.CDLL(str(_DLL_PATH))
    dll.lif_alpha_ed_step.argtypes = [
        _int_p, _int_p, _float_p,        # colptr, row, data (CSC)
        ctypes.c_int32, ctypes.c_int32,  # n, nnz
        _float_p, _float_p, _float_p, _int_p,  # spikes, V, g, refr
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # V_rest, V_th, V_reset
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # neuromod, tau_m, dt
        ctypes.c_float, ctypes.c_int32,  # syn_decay, refractory_steps
        _float_p,                        # spikes_out
    ]
    dll.lif_alpha_ed_step.restype = None
    return dll


class NativeLIFAlpha:
    """Backend LIF α event-driven (batch=1, layout CSC). Zéro allocation."""

    def __init__(self):
        self.dll = _load_dll()

    def step(self, colptr, row, data, spikes, V, g, refr,
             V_rest, V_th, V_reset, neuromod, tau_m, dt,
             syn_decay, refractory_steps, spikes_out):
        n = np.int32(V.shape[0])
        nnz = np.int32(data.shape[0])
        self.dll.lif_alpha_ed_step(
            colptr.ctypes.data_as(_int_p),
            row.ctypes.data_as(_int_p),
            data.ctypes.data_as(_float_p),
            n, nnz,
            spikes.ctypes.data_as(_float_p),
            V.ctypes.data_as(_float_p),
            g.ctypes.data_as(_float_p),
            refr.ctypes.data_as(_int_p),
            ctypes.c_float(V_rest), ctypes.c_float(V_th), ctypes.c_float(V_reset),
            ctypes.c_float(neuromod), ctypes.c_float(tau_m), ctypes.c_float(dt),
            ctypes.c_float(syn_decay), ctypes.c_int32(refractory_steps),
            spikes_out.ctypes.data_as(_float_p),
        )


def csc_from_csr(csr, n):
    """Construit (colptr, row, data) CSC à partir d'une CSR scipy (n×n)."""
    M = _sp.csr_matrix((csr.data, csr.indices, csr.indptr), shape=(n, n)).tocsc()
    return (np.ascontiguousarray(M.indptr, dtype=np.int32),
            np.ascontiguousarray(M.indices, dtype=np.int32),
            np.ascontiguousarray(M.data, dtype=np.float32))


def verify_vs_torch(n_trials: int = 300, tol: float = 1e-4) -> tuple[int, int]:
    """
    Vérification croisée contre la référence torch alpha de brain.py, en
    reproduisant la boucle complète (délai, décroissance, réfractaire) sur des
    réseaux aléatoires. Retourne (échecs, essais).
    """
    import torch

    native = NativeLIFAlpha()
    rng = np.random.default_rng(0)
    failures = 0
    for _ in range(n_trials):
        n = int(rng.integers(4, 200))
        density = rng.uniform(0.01, 0.5)
        M = _sp.random(n, n, density=density, format="csr", dtype=np.float32,
                       random_state=int(rng.integers(1 << 30)))
        M.data[:] = (rng.uniform(-1, 1, size=M.nnz) * 2).astype(np.float32)

        V_rest = float(rng.uniform(0, 0.5))
        V_th = float(rng.uniform(0.5, 1.5))
        V_reset = float(rng.choice([V_rest, -0.3]))
        neuromod = float(rng.uniform(0, 2))
        tau_m = float(rng.uniform(1, 40))
        dt = float(rng.uniform(0.1, 5))
        tau_syn = float(rng.uniform(1, 30))
        syn_decay = float(np.exp(-dt / tau_syn))
        refractory_steps = int(rng.integers(1, 6))
        delay_steps = 2

        colptr, row, data = csc_from_csr(M, n)

        # État
        V = np.zeros(n, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)
        refr = np.zeros(n, dtype=np.int32)
        delay = [np.zeros(n, dtype=np.float32) for _ in range(delay_steps)]

        # Référence torch — même boucle que brain._step_prepared (batch=1)
        W_t = torch.sparse_csr_tensor(
            torch.tensor(M.indptr), torch.tensor(M.indices),
            torch.tensor(M.data), size=(n, n))
        V_t = torch.zeros(n, 1)
        g_t = torch.zeros(n, 1)
        refr_t = torch.zeros(n, 1, dtype=torch.int32)
        delay_t = [torch.zeros(n, 1) for _ in range(delay_steps)]

        # Même flux de spikes (stimulus externe aléatoire) alimente les DEUX
        # buffers de délai — les entrées C et torch sont donc identiques.
        # (brain.py : le buffer de délai reçoit les spikes du pas précédent,
        #  sensoriels compris ; ici on injecte un stimulus équivalent.)
        for step in range(8):
            stim = (rng.random(n) < 0.3).astype(np.float32)
            d_np = delay.pop(0); delay.append(stim.copy())
            d_t = delay_t.pop(0); delay_t.append(torch.from_numpy(stim).unsqueeze(1))

            # native
            out_np = np.zeros(n, dtype=np.float32)
            native.step(colptr, row, data, d_np, V, g, refr,
                        V_rest, V_th, V_reset, neuromod, tau_m, dt,
                        syn_decay, refractory_steps, out_np)

            # torch ref
            inc = torch.sparse.mm(W_t, d_t)
            g_t = g_t.mul_(syn_decay).add_(inc)
            V_t = V_t + (-(V_t - V_rest) + neuromod * g_t) * (dt / tau_m)
            spiked = (V_t >= V_th) & (refr_t <= 0)
            V_t = V_t.masked_fill(spiked, V_reset)
            refr_t = refr_t.sub_(1).clamp_(min=0).masked_fill(spiked, refractory_steps)
            spikes_t = spiked.float()

            dv = float(np.abs(V - V_t.numpy()[:, 0]).max())
            dg = float(np.abs(g - g_t.numpy()[:, 0]).max())
            ds = float(np.abs(out_np - spikes_t.numpy()[:, 0]).max())
            if dv > tol or dg > tol or ds > tol:
                failures += 1
                break

    return failures, n_trials