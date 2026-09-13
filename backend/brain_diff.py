"""
Simulation différentiable minimale (RAPPORT B3, phase optionnelle).

Idée : entraîner l'ENCODAGE (projection aléatoire → apprise) en traversant
UN pas LIF avec surrogate gradient (spike ≈ sigmoid(β·(V−Vth))), connectome
gelé. Full BPTT × 60 pas × 26M synapses sur CPU = heures/époque : ce module
prouve le mécanisme sur petit réseau, pas la recette d'entraînement.

Usage : python brain_diff.py  (self-check <10 s, grad ≠ 0 vers l'encodeur)
"""
import torch


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, V_th: float = 1.0, beta: float = 20.0):
        ctx.save_for_backward(V)
        ctx.beta, ctx.V_th = beta, V_th
        return (V >= V_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (V,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.beta * (V - ctx.V_th))
        return grad_out * sig * (1 - sig) * ctx.beta, None, None


def lif_step_diff(V, spikes_in, W, tau_m: float = 20.0, dt: float = 1.0,
                  V_th: float = 1.0, V_reset: float = 0.0, I_ext=None):
    """Un pas LIF différentiable (W gelé côté appelant via detach si besoin)."""
    I_syn = spikes_in @ W.T
    if I_ext is not None:
        I_syn = I_syn + I_ext
    V = V + (-V + I_syn) / tau_m * dt
    sp = SurrogateSpike.apply(V, V_th, 20.0)
    V = torch.where(sp > 0.5, torch.full_like(V, V_reset), V)
    return V, sp


def demo(n: int = 64, feats: int = 16, seed: int = 0):
    """L'encodeur apprend : grad du loss (taux moteur − cible) remonte à proj."""
    torch.manual_seed(seed)
    W = torch.randn(n, n).detach() * 0.05  # câblage gelé
    proj = torch.randn(feats, n, requires_grad=True)  # ← CE qu'on entraînerait
    V = torch.zeros(1, n)
    x = torch.randn(1, feats)
    I_ext = x @ proj
    for _ in range(5):
        V, sp = lif_step_diff(V, torch.zeros(1, n), W, I_ext=I_ext * 0.2)
    loss = (sp[:, -8:].mean() - 0.3) ** 2
    loss.backward()
    return float(loss.item()), float(proj.grad.abs().sum().item())


if __name__ == "__main__":
    loss, g = demo()
    assert g > 0, "le gradient doit atteindre l'encodeur"
    print(f"[diff] OK : loss={loss:.4f}, |grad encodeur|={g:.4f} > 0")
