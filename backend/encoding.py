"""
Encodage du plateau d'échecs en courants injectés dans les neurones sensoriels.
Inspiré de doomfly (frames → 3335 entrées de luminosité + 811 de couleur).
"""
import numpy as np
import torch
import chess

# Mapping type de pièce → intensité
PIECE_INTENSITY = {
    chess.PAWN: 0.5,
    chess.KNIGHT: 0.8,
    chess.BISHOP: 0.8,
    chess.ROOK: 1.0,
    chess.QUEEN: 1.5,
    chess.KING: 2.0,
}

# Gain du stimulus : sans lui, les courants restent sub-seuil (V_th=1.0) et
# AUCUN neurone ne spike pendant l'évaluation des coups (calibré ×20 pour un
# régime de décharge moteur ~0.3 spikes/pas enregistré).
STIM_GAIN = 20.0

# Normalisation à norme constante : diagnostic SVD montré nécessaire —
# sans elle, la réponse du réseau est ~rank-1 (un simple niveau d'activité
# global, parfois nul) et AUCUN readout linéaire ne peut discriminer les
# positions. À norme fixe, l'information passe par le PATTERN d'activation.
# Cible = norme typique des courants APRÈS gain (≈ 22.7 × 20) : ni sub-seuil,
# ni saturation.
STIM_TARGET_NORM = 450.0


class BoardEncoder:
    """
    Encode un échiquier en courants pour les neurones sensoriels.

    Schéma :
      - 64 cases × 12 types de pièces (6 types × 2 couleurs) = 768 features
      - On projette ces features sur n_sensory neurones via une matrice aléatoire fixe.
      - Les features incluent aussi : nombre de coups légaux, roi en échec, etc.
    """

    def __init__(self, n_sensory: int, seed: int = 42, extra_features: int = 20):
        self.n_sensory = n_sensory
        self.n_features = 64 * 12 + extra_features
        rng = np.random.default_rng(seed)
        # Projection aléatoire fixe (n_features → n_sensory)
        self.proj = rng.normal(
            scale=1.0 / np.sqrt(self.n_features), size=(self.n_features, n_sensory)
        ).astype(np.float32)

    def _features(self, board: chess.Board) -> np.ndarray:
        """Vecteur de features (n_features,) d'une position."""
        features = np.zeros(self.n_features, dtype=np.float32)

        # 64 cases × 12 types
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
                features[square * 12 + piece_idx] = PIECE_INTENSITY[piece.piece_type]

        # Features extra
        offset = 64 * 12
        features[offset + 0] = len(list(board.legal_moves)) / 40.0
        features[offset + 1] = float(board.is_check())
        features[offset + 2] = float(board.is_checkmate())
        features[offset + 3] = float(board.is_stalemate())
        features[offset + 4] = float(board.turn)
        features[offset + 5] = board.fullmove_number / 100.0
        # Matériel restant (différence blancs - noirs, normalisée)
        material = sum(
            (1 if p.piece_type == chess.PAWN else
             3 if p.piece_type in (chess.KNIGHT, chess.BISHOP) else
             5 if p.piece_type == chess.ROOK else
             9 if p.piece_type == chess.QUEEN else 0)
            * (1 if p.color else -1)
            for p in board.piece_map().values()
        )
        features[offset + 6] = material / 40.0
        # ... autres features laissées à 0
        return features

    def encode(self, board: chess.Board) -> torch.Tensor:
        """Retourne un vecteur (n_sensory,) de courants, norme constante."""
        currents = self._features(board) @ self.proj
        c = torch.tensor(currents, dtype=torch.float32) * STIM_GAIN
        n = float(c.norm())
        return c * (STIM_TARGET_NORM / n) if n > 0 else c

    def encode_batch(self, boards: list[chess.Board]) -> torch.Tensor:
        """Retourne (n_sensory, B) : les courants de B positions en un seul
        batch, chaque colonne normalisée à norme constante."""
        feats = np.stack([self._features(b) for b in boards], axis=0)  # (B, n_features)
        currents = feats @ self.proj  # (B, n_sensory)
        c = torch.tensor(currents.T, dtype=torch.float32) * STIM_GAIN  # (n_sensory, B)
        norms = c.norm(dim=0, keepdim=True)
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        return c * (STIM_TARGET_NORM / norms)


# --- Encodage modal (RAPPORT A2) : 4 canaux biologiquement motivés ---
# Chaque sens reçoit une facette de la position au lieu d'une projection
# globale. Testable : même norme totale 450, mais variance inter-positions
# et rang SVD supérieurs (le pattern remplace le niveau global).
MODAL_CHANNELS = ("visual", "mechano", "gustatory", "olfactory")

class ModalBoardEncoder(BoardEncoder):
    """Variante structurée de BoardEncoder : n_sensory découpé en 4 blocs,
    un par modalité. Projection aléatoire conservée *à l'intérieur* de chaque
    canal (norme par canal = 450/sqrt(4) → norme totale 450). Opt-in :
    BoardEncoder reste le défaut pour non-régression Shiu."""

    def __init__(self, n_sensory: int, seed: int = 42, extra_features: int = 20):
        self.n_sensory = n_sensory
        self.n_features = 64 * 12 + extra_features
        # Découpage sensoriel en 4 blocs quasi égaux
        base, rem = divmod(n_sensory, 4)
        self.block_sizes = [base + (1 if i < rem else 0) for i in range(4)]
        self.block_offsets = np.cumsum([0] + self.block_sizes)
        # Découpage features : pièces / menaces-contacts / matériel / contexte roi
        # 0:768 pièces (visuel), 768:773 menaces+échec (mécano),
        # 773:775 matériel+trait (gustatif), 775:788 contexte (olfactif)
        self.feat_slices = [(0, 768), (768, 773), (773, 775), (775, self.n_features)]
        rng = np.random.default_rng(seed)
        self.projs = []
        for (a, b), nb in zip(self.feat_slices, self.block_sizes):
            nf = b - a
            self.projs.append(rng.normal(
                scale=1.0 / np.sqrt(max(nf, 1)), size=(nf, max(nb, 1))).astype(np.float32))
        self.per_canal_norm = STIM_TARGET_NORM / np.sqrt(4)

    def _encode_feats(self, feats: np.ndarray) -> np.ndarray:
        """feats (B, n_features) → currents (B, n_sensory), norme par canal."""
        parts = []
        for (a, b), proj, nb in zip(self.feat_slices, self.projs, self.block_sizes):
            if nb == 0:
                continue
            cur = feats[:, a:b] @ proj  # (B, nb)
            c = cur * STIM_GAIN
            n = np.linalg.norm(c, axis=1, keepdims=True)
            # Canal vide (ex. matériel nul en début de partie) → reste à 0,
            # les autres canaux gardent leur norme ; renorm globale après.
            scale = np.where(n > 1e-9, self.per_canal_norm / np.maximum(n, 1e-9), 0.0)
            parts.append(c * scale)
        out = np.concatenate(parts, axis=1) if parts else np.zeros((len(feats), 0), np.float32)
        # Norme totale 450 garantie (canaux actifs mis à l'échelle ensemble)
        tot = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(tot > 1e-9, out * (STIM_TARGET_NORM / np.maximum(tot, 1e-9)), out)
        return out.astype(np.float32)

    def encode(self, board: chess.Board) -> torch.Tensor:
        feats = self._features(board)[None, :]
        return torch.tensor(self._encode_feats(feats).T, dtype=torch.float32).squeeze(1)

    def encode_batch(self, boards: list[chess.Board]) -> torch.Tensor:
        feats = np.stack([self._features(b) for b in boards], axis=0)
        return torch.tensor(self._encode_feats(feats).T, dtype=torch.float32)
