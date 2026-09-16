"""
flyintel — brain of the fly, as a reusable, trainable module.

Single entry point for everything that needs the fly's connectome brain
without depending on the web backend. It re-exports the battle-tested
components that already live in backend/ and adds:

  - a clean public API (load -> build -> train -> evaluate)
  - an optional SpearVM-accelerated backend (spur-math kernels)
  - benchmark runners that score the brain on several domains
  - a JSON leaderboard

No code is duplicated here on purpose (ponytail: reuse over rewrite). This
package is a thin, importable facade + the genuinely new accelerations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/ is not a package; add it to the import path so we can reuse its
# modules verbatim instead of forking them.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from brain import FlyBrain, build_prod_brain          # noqa: E402
from readout import LinearReadout, load_readout        # noqa: E402
from readout_looped import (                          # noqa: E402
    LoopedReadout,
    trajectory_from_brain,
    save_looped,
    load_looped,
)
from encoding import BoardEncoder, ModalBoardEncoder, AnythingEncoder   # noqa: E402
from data_loader import (                             # noqa: E402
    load_connectome,
    dn_maxflow_mask,
    DATA_DIR,
    NPZ_PATH,
)

__version__ = "0.1.0"

# Sous-modules accélérés (importés paresseusement pour éviter torch au top-level
# quand on ne veut que charger le brain)
from . import backends  # noqa: E402  (benchmark_backends, gelu_fast, tanh_fast)
from . import bench  # noqa: E402     (run_all, summary, leaderboard)
from . import train  # noqa: E402     (train_looped)
from . import spear_math  # noqa: E402  (formules champions superspear, benchmarkées)
from . import readouts_spear  # noqa: E402  (readouts à activations Spear)
from . import websearch  # noqa: E402  (search/learn web — Exa ou DuckDuckGo)

__all__ = [
    "FlyBrain",
    "build_prod_brain",
    "LinearReadout",
    "load_readout",
    "LoopedReadout",
    "trajectory_from_brain",
    "save_looped",
    "load_looped",
    "BoardEncoder",
    "ModalBoardEncoder",
    "AnythingEncoder",
    "load_connectome",
    "dn_maxflow_mask",
    "DATA_DIR",
    "NPZ_PATH",
    "has_malecns",
    "default_device",
    "load_brain",
    "load_readout_module",
]


def has_malecns() -> bool:
    """True si le connectome MaleCNS réel est présent sur disque."""
    return NPZ_PATH.exists()


def default_device() -> str:
    """cuda si dispo (GPU torch), sinon cpu. Overridable via FLY_DEVICE."""
    import torch

    env = os.environ.get("FLY_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_brain(
    force_synthetic: bool = False,
    synapse_model: str | None = None,
    wsyn: float | None = None,
    **kwargs,
):
    """Charge le connectome et construit un cerveau prêt à l'emploi.

    Retourne (brain, conn). Sur MaleCNS présent → régime prod alpha
    (Shiu). Sinon → connectome synthétique (démo) en mode current.
    """
    conn = load_connectome(force_synthetic=force_synthetic)
    if conn.get("W").shape[0] > 100_000:
        brain = build_prod_brain(conn, synapse_model=synapse_model, wsyn=wsyn)
    else:
        brain = FlyBrain(
            conn["W"], conn["is_sensory"], conn["is_motor"],
            conn["is_descending"], **kwargs,
        )
    return brain, conn


def load_readout_module(path: str, brain, conn, encoder_name: str = "classic",
                        dn_idx=None, device: str | None = None):
    """Charge un readout (linéaire ou looped) déjà entraîné.

    Détecte le format au checkpoint (state_dict keys). Retourne
    (readout, meta) où meta a la forme du dict de save_looped.
    """
    import torch

    device = device or default_device()
    n_motor = int(conn["is_motor"].sum())
    n_desc = int(dn_idx.sum()) if dn_idx is not None else int(conn["is_descending"].sum())

    blob = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in blob and any(k.startswith("cell.") for k in blob["state_dict"]):
        model, meta = load_looped(path, device)
        if dn_idx is not None:
            meta["dn_idx"] = dn_idx
        return model, meta
    # Linéaire
    model = load_readout(path, n_motor, n_desc, device)
    return model, {"encoder": encoder_name}