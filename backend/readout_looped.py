"""
LoopedReadout (RAPPORT B2) : décodeur à profondeur temporelle au-dessus du
connectome LIF gelé. Même objet en train et inférence (principe RLT).

  x_t (frame lue au pas t) → Merge(e_t, s_{t-1}) → GRU → SWA(16) → score

Usage minimal :
  dec = LoopedReadout(d_in=n_motor+n_desc)
  scores_T = dec(traj)          # traj (T, B, d_in) ex. motor_hist
  score_final = scores_T[-1]    # supervision au pas T
"""
import numpy as np
import torch
import torch.nn as nn


class LoopedCell(nn.Module):
    def __init__(self, d_in: int, d_h: int = 128, window: int = 16):
        super().__init__()
        self.d_h = d_h
        self.window = window
        self.merge = nn.Linear(d_in + d_h, d_h)
        self.gru = nn.GRUCell(d_h, d_h)
        self.swa = nn.Linear(window * d_h, d_h)
        self.head = nn.Linear(d_h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (T, B, d_in) → scores (T, B). s_0=0, pas de reset entre positions
        si l'appelant garde l'état (ici reset par forward, persistance = caller)."""
        T, B, _ = x.shape
        s = x.new_zeros(B, self.d_h)
        buf: list[torch.Tensor] = []
        outs = []
        for t in range(T):
            h = torch.tanh(self.merge(torch.cat([x[t], s], dim=-1)))
            s = self.gru(h, s)
            buf.append(s)
            w = torch.stack(buf[-self.window:], dim=1).flatten(1)
            if len(buf) < self.window:
                pad = self.d_h * (self.window - len(buf))
                w = torch.cat([s.new_zeros(B, pad), w], dim=1)
            outs.append(self.head(torch.tanh(self.swa(w))).squeeze(-1))
        return torch.stack(outs, dim=0)  # (T, B)


class LoopedReadout(nn.Module):
    """Enveloppe : normalise chaque frame comme LinearReadout puis décode."""

    def __init__(self, d_in: int, d_h: int = 128, window: int = 16):
        super().__init__()
        self.cell = LoopedCell(d_in, d_h, window)
        self.register_buffer("baseline", torch.zeros(d_in))

    def observe(self, x: torch.Tensor, momentum: float = 0.05) -> None:
        with torch.no_grad():
            mean = x.reshape(-1, x.shape[-1]).mean(dim=0)
            if self.baseline.abs().sum() == 0:
                self.baseline.copy_(mean)
            else:
                self.baseline.mul_(1.0 - momentum).add_(mean, alpha=momentum)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """traj (T, B, d_in) → scores (T, B)."""
        xc = traj - self.baseline
        xn = xc / (xc.norm(dim=-1, keepdim=True) + 1e-6)
        return self.cell(xn)


def trajectory_from_brain(result: dict) -> torch.Tensor:
    """Construit (T, B, d_in) depuis brain.run(return_history=True).
    Concatène moteur + descendant le long de d_in."""
    mh = result["motor_hist"]          # (T, B, n_motor)
    dh = result.get("descending_hist")  # (T, B, n_desc) | None
    if dh is None:
        return mh
    return torch.cat([mh, dh], dim=-1)


def save_looped(path: str, model: "LoopedReadout", dn_mask=None, encoder: str = "classic"):
    """Checkpoint + méta (même objet train/inférence, principe RLT B1).
    100 % tenseurs/strings : chargeable en weights_only=True (torch ≥ 2.6)."""
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": int(model.cell.merge.in_features - model.cell.d_h),
        "d_h": int(model.cell.d_h),
        "window": int(model.cell.window),
        "encoder": str(encoder),
        "dn_mask": torch.as_tensor(np.asarray(dn_mask, dtype=bool)) if dn_mask is not None else None,
    }, path)


def load_looped(path: str, device: str):
    """Charge un LoopedReadout + ses métas. Retourne (model, meta).
    meta["dn_mask"] : np.ndarray booléen ou None."""
    import numpy as _np
    blob = torch.load(path, map_location=device, weights_only=True)
    sd = blob["state_dict"]
    model = LoopedReadout(int(blob["d_in"]), d_h=int(blob["d_h"]),
                          window=int(blob["window"])).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    meta = {"d_in": int(blob["d_in"]), "d_h": int(blob["d_h"]),
            "window": int(blob["window"]), "encoder": str(blob.get("encoder", "classic")),
            "dn_mask": (_np.asarray(blob["dn_mask"].cpu(), dtype=bool)
                        if blob.get("dn_mask") is not None else None)}
    return model, meta


if __name__ == "__main__":
    # Self-check runnable : python readout_looped.py (ponytail: un assert-test, pas de framework)
    torch.manual_seed(0)
    dec = LoopedReadout(d_in=32, d_h=16, window=4)
    traj = torch.randn(6, 3, 32)
    out = dec(traj)
    assert out.shape == (6, 3), out.shape
    out[-1].sum().backward()  # BPTT passe (connectome gelé côté appelant)
    assert dec.cell.gru.weight_hh.grad is not None
    print("[looped] OK : (T,B)=(6,3) -> scores (6,3), grad OK")
