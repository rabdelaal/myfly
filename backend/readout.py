"""
Readout linéaire entraîné : activité motrice → score du coup.

Décodage par vecteur populationnel CENTRÉ : la réponse du connectome contient
~94 % de composante commune (recrutement global indépendant de la position,
mesuré empiriquement). On soustrait une baseline (moyenne mobile des réponses,
persistée dans le checkpoint) pour que le readout n'apprenne que la partie
discriminante du pattern neuronal.
"""
import torch
import torch.nn as nn


class LinearReadout(nn.Module):
    def __init__(self, n_motor: int, n_descending: int):
        super().__init__()
        self.n_in = n_motor + n_descending
        # Sortie : un score scalaire
        self.fc = nn.Linear(self.n_in, 1)
        # Baseline de réponse (moyenne mobile), persistée avec le modèle
        self.register_buffer("baseline", torch.zeros(self.n_in))

    def observe(self, x: torch.Tensor, momentum: float = 0.05) -> None:
        """Met à jour la baseline avec la réponse courante (batch ou vecteur)."""
        with torch.no_grad():
            mean = x.mean(dim=0)
            if self.baseline.abs().sum() == 0:
                self.baseline.copy_(mean)
            else:
                self.baseline.mul_(1.0 - momentum).add_(mean, alpha=momentum)

    def forward(self, motor: torch.Tensor, descending: torch.Tensor) -> torch.Tensor:
        x = torch.cat([motor, descending], dim=-1) - self.baseline
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return self.fc(x).squeeze(-1)


def load_readout(path: str, n_motor: int, n_descending: int, device: str):
    model = LinearReadout(n_motor, n_descending).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)  # baseline absente → zéros
    model.eval()
    return model
