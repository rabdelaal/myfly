"""
Logique d'échecs : génération des coups, évaluation par la mouche, choix.
Scoring batché : tous les coups candidats sont simulés en PARALLÈLE dans le
connectome (états neuronaux (n, B)), au lieu de B simulations séquentielles.
"""
import chess
import torch
import numpy as np


class FlyChessEngine:
    def __init__(self, brain, encoder, readout, max_candidates: int = 20,
                 n_steps: int = 300, noise: float = 0.0, dn_mask=None):
        self.brain = brain
        self.encoder = encoder
        self.readout = readout
        # Sous-ensemble DN max-flow (A3) : masque booléen sur les descendants, ou None
        self.dn_mask = dn_mask
        self._last_motor_means = None   # (B,) après chaque _score_batch
        self._last_winner_motor = None  # scalaire après choose/sample_move
        self.max_candidates = max_candidates
        self.n_steps = n_steps
        # Bruit ajouté aux scores : une mouche affamée/épuisée joue moins bien.
        self.noise = noise
        # Adaptatif à l'échelle : sur le vrai MaleCNS (166k neurones), réduire
        # la profondeur de simulation pour garder un temps de réflexion jouable.
        if brain.n > 100_000:
            self.n_steps = min(n_steps, 60)
            self.max_candidates = min(max_candidates, 8)
        # En mode alpha, la propagation est plus rapide (τ_syn = 5 ms) mais le
        # drive Poisson calibré (norme 450, fix x20) met ~100 pas à recruter
        # les moteurs sur MaleCNS (5 spikes à 40 pas → 865 à 100 pas, mesuré).
        # Le batch B candidats reste parallèle : ~7 s/coup à 73 ms/pas.
        if getattr(brain, "synapse_model", "current") == "alpha":
            self.n_steps = min(self.n_steps, 100)

    def _candidates(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]
        others = [m for m in legal_moves if not board.is_capture(m)]
        ordered = captures + others
        return ordered[: self.max_candidates], captures

    def _is_looped(self) -> bool:
        return hasattr(self.readout, "cell")  # LoopedReadout vs LinearReadout

    def _slice_dn(self, traj):
        """Restreint la partie descendante de traj (T,B,motor+desc) au masque max-flow."""
        if self.dn_mask is None:
            return traj
        import torch as _t
        motor_n = int(self.brain.is_motor.sum())
        keep = np.r_[np.ones(motor_n, bool), np.asarray(self.dn_mask, dtype=bool)]
        return traj[..., _t.tensor(keep, device=traj.device)]

    def _score_batch(self, boards: list[chess.Board]) -> tuple[np.ndarray, list | None]:
        """Simule la mouche sur B plateaux EN PARALLÈLE.
        Retourne (scores, frames_all[T][B]) — les frames servent au replay
        du coup gagnant SANS re-simuler (une 2e simu complète économisée).
        Expose aussi self._last_motor_means (B,) : activité motrice moyenne
        par candidat — le pet s'en sert comme réaction (0 simu de plus)."""
        # NOTE: encode_*() applique déjà STIM_GAIN + norme 450 — ne pas re-multiplier
        # (ancien bug ×20 → saturation, corrigé ; voir test_encoding_norm).
        currents = self.encoder.encode_batch(boards).to(self.brain.device)
        collect = True  # toujours : coût ~80ms, le replay est gratuit derrière
        if self._is_looped():
            from readout_looped import trajectory_from_brain
            with torch.no_grad():
                result = self.brain.run(currents, n_steps=self.n_steps,
                                        record_every=10, return_history=True,
                                        collect_frames=collect)
                traj = self._slice_dn(trajectory_from_brain(result))
                self.readout.observe(traj.reshape(-1, traj.shape[-1]))
                scores = self.readout(traj)[-1]  # supervision au pas T (B,)
        else:
            result = self.brain.run(currents, n_steps=self.n_steps, record_every=10,
                                    collect_frames=collect)
            motor = result["motor_mean"]            # (B, n_motor) sur device
            descending = result["descending_last"]  # (B, n_desc) sur device
            if self.dn_mask is not None:
                descending = descending[:, torch.tensor(
                    np.asarray(self.dn_mask, dtype=bool), device=descending.device)]
            with torch.no_grad():
                # Le forward du readout centre (baseline) et normalise le pattern
                self.readout.observe(torch.cat([motor, descending], dim=-1))
                scores = self.readout(motor, descending)  # (B,)
        scores = scores.cpu().numpy()
        if self.noise > 0:
            scores = scores + np.random.normal(0.0, self.noise, size=scores.shape)
        self._last_motor_means = result["motor_mean"].mean(dim=1).detach().cpu()
        return scores, result.get("frames_all")

    def choose_move(self, board: chess.Board, frame_callback=None):
        """
        Retourne (meilleur_coup, scores_dict).
        Règles "assist" : prendre si possible. Les frames de la simulation du
        meilleur coup sont streamées via frame_callback (ex. WebSocket) si fourni.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        candidates, captures = self._candidates(board)

        # Un plateau par candidat, évalués en un seul batch
        boards = []
        for move in candidates:
            board.push(move)
            boards.append(board.copy())
            board.pop()

        scores_arr, frames_all = self._score_batch(boards)
        scores = {move.uci(): float(s) for move, s in zip(candidates, scores_arr)}

        # Règle assist 1 : prendre si possible
        if captures:
            capture_moves = [m for m in candidates if board.is_capture(m)]
            best_capture = max(capture_moves, key=lambda m: scores[m.uci()])
            best_move = best_capture
        else:
            best_move = max(candidates, key=lambda m: scores[m.uci()])

        # Replay SANS re-simuler : les frames du gagnant viennent du batch
        # (même stimulus, mêmes spikes — envoi en rafale, le front anime).
        # L'activité motrice du gagnant sert de réaction au pet (0 simu de plus).
        col = candidates.index(best_move)
        self._last_winner_motor = float(self._last_motor_means[col])
        if frame_callback is not None and frames_all is not None:
            for frame_cols in frames_all:
                frame_callback(frame_cols[col])
        return best_move, scores

    def sample_move(self, board: chess.Board, temperature: float = 1.0):
        """Politique explicite (RAPPORT B4) : softmax(scores / T) → coup échantillonné.

        T asservie à l'octopamine côté appelant (bruit d'humeur = exploration).
        La simulation LIF étant déterministe à stimulus fixé, re-simuler sous
        les poids courants = « exact current-policy replay » sans importance
        sampling. Retourne (coup, scores_dict)."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}
        candidates, _ = self._candidates(board)
        boards = []
        for move in candidates:
            board.push(move)
            boards.append(board.copy())
            board.pop()
        arr = self._score_batch(boards)[0]
        t = max(float(temperature), 1e-3)
        z = arr - arr.max()
        probs = np.exp(z / t)
        probs = probs / probs.sum()
        best_move = np.random.choice(candidates, p=probs)
        self._last_winner_motor = float(self._last_motor_means[list(candidates).index(best_move)])
        return best_move, {m.uci(): float(s) for m, s in zip(candidates, arr)}
