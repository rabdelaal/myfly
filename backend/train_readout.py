"""
Entraîne le readout linéaire avec une règle locale à trois facteurs :
    Δw = lr * erreur * activité_entrée
C'est la version bioplastique de la descente de gradient pour une couche
linéaire — la plasticité est purement locale (pas de rétropropagation à
travers la simulation du connectome), la récompense jouant le rôle de la
dopamine. Récompense = score Stockfish de la position.

Usage : python train_readout.py --episodes 200
"""
import argparse
import os
import shutil
import time
from pathlib import Path
import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
from readout import LinearReadout


def make_val_set(n_positions: int, seed: int = 77):
    """Positions de validation (jamais vues pendant l'entraînement)."""
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n_positions):
        b = chess.Board()
        for _ in range(rng.integers(1, 30)):
            if b.is_game_over():
                break
            b.push(np.random.choice(list(b.legal_moves)))
        boards.append(b)
    return boards


def validate(readout, val_set, brain, encoder, steps) -> float | None:
    """rho de Spearman, validation BATCHÉE en un seul run (B=n)."""
    from scipy.stats import spearmanr
    boards = [b for b, _ in val_set]
    targets = [t for _, t in val_set]
    with torch.no_grad():
        # encode() inclut déjà STIM_GAIN + norme 450
        currents = encoder.encode_batch(boards).to(brain.device)
        r = brain.run(currents, n_steps=steps, record_every=10)
        preds = readout(r["motor_mean"], r["descending_last"]).cpu().tolist()
    if len(set(np.round(preds, 6))) < 3:
        return None
    return float(spearmanr(targets, preds).correlation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=60)
    parser.add_argument("--output", type=str, default="readout.pt")
    parser.add_argument("--stockfish", type=str, default="stockfish")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="positions par run connectome (B) : ~4x plus vite à 8")
    parser.add_argument("--steps", type=int, default=200,
                        help="pas de simulation LIF par position "
                             "(réduire sur le vrai MaleCNS : ~73 ms/pas)")
    args = parser.parse_args()

    # Garde-fou « jamais de train long » : ~5 s/position sur MaleCNS (mesuré).
    # Au-delà du budget, refus sauf opt-in explicite.
    budget = int(os.environ.get("FLY_TRAIN_BUDGET", "120"))
    if args.episodes * args.max_moves > budget \
            and os.environ.get("FLY_ALLOW_LONG_TRAIN") != "1":
        raise SystemExit(
            f"[train] refusé : {args.episodes * args.max_moves} positions "
            f"> budget {budget}. Opt-in : FLY_ALLOW_LONG_TRAIN=1, ou "
            f"FLY_TRAIN_BUDGET=N.")

    print("[train] Chargement du connectome...")
    conn = load_connectome()
    from brain import build_prod_brain
    brain = build_prod_brain(conn)  # régime prod alpha (current = silence sur MaleCNS)
    encoder = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))

    n_motor = int(conn["is_motor"].sum())
    n_desc = int(conn["is_descending"].sum())
    readout = LinearReadout(n_motor, n_desc).to(brain.device)

    # Un seul moteur Stockfish pour toute la session (signal de récompense)
    # Retry : sous pression mémoire (connectome chargé), le handshake UCI
    # peut dépasser le timeout interne de python-chess.
    stockfish = None
    for attempt in range(5):
        try:
            stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            break
        except Exception as e:
            print(f"[train] Stockfish tentative {attempt + 1}/5 échouée : {e}")
            time.sleep(5)
    if stockfish is None:
        raise SystemExit("[train] Stockfish introuvable ou trop lent à démarrer")

    def reward_of(board: chess.Board) -> float:
        try:
            info = stockfish.analyse(board, chess.engine.Limit(depth=args.depth))
            return info["score"].white().score(mate_score=10000) / 1000.0
        except Exception:
            return 0.0

    lr = args.lr
    losses = []
    total_positions = 0
    best_val = -1.0
    best_path = args.output + ".best"

    # Positions de validation tenues à l'écart, scorées une fois pour toutes
    val_set = [(b, reward_of(b)) for b in make_val_set(15)]
    print(f"[train] {len(val_set)} positions de validation tenues à l'écart")

    def train_batch(boards_b, targets_b, total_positions):
        """Mini-batch : 1 seul run connectome pour K positions (B=K).
        Mise à jour = moyenne des mises à jour 3-facteurs (même point fixe)."""
        nonlocal best_val
        prev = total_positions
        # encode() inclut déjà STIM_GAIN + norme 450
        currents = encoder.encode_batch(boards_b).to(brain.device)
        with torch.no_grad():
            result = brain.run(currents, n_steps=args.steps, record_every=10)
            x = torch.cat([result["motor_mean"], result["descending_last"]], dim=-1)
            readout.observe(x)
            xc = x - readout.baseline
            xn = xc / (xc.norm(dim=-1, keepdim=True) + 1e-6)
            pred = readout.fc(xn).squeeze(-1)
            err = torch.tensor(targets_b, device=brain.device) - pred
            readout.fc.weight.add_(lr * (err.unsqueeze(1) * xn).mean(0, keepdim=True))
            readout.fc.bias.add_(lr * err.mean().unsqueeze(0))
            readout.fc.weight.clamp_(-10.0, 10.0)
            readout.fc.bias.clamp_(-10.0, 10.0)
            losses.extend((err * err).cpu().tolist())
        total_positions += len(boards_b)
        if prev // 20 != total_positions // 20:
            vr = validate(readout, val_set, brain, encoder, args.steps)
            if vr is not None and vr > best_val:
                best_val = vr
                torch.save(readout.state_dict(), best_path)
                tqdm.write(f"  {total_positions} pos | val ρ = {vr:+.3f} "
                           f"★ meilleur → sauvegardé")
        if prev // 10 != total_positions // 10:
            rms = float(np.sqrt(np.mean(losses[-50:])))
            rate = (time.time() - t_start) / total_positions
            eta = rate * (args.episodes * args.max_moves - total_positions)
            tqdm.write(f"  {total_positions} positions | RMS récent {rms:.3f} | "
                       f"{rate:.1f} s/position | ETA {eta/60:.0f} min")
        return total_positions

    t_start = time.time()
    try:
        for episode in tqdm(range(args.episodes), desc="Épisodes"):
            board = chess.Board()
            buf_b, buf_t = [], []
            for _ in range(args.max_moves):
                if board.is_game_over():
                    break
                move = np.random.choice(list(board.legal_moves))
                board.push(move)
                buf_b.append(board.copy())
                buf_t.append(reward_of(board))
                if len(buf_b) >= args.batch_size:
                    total_positions = train_batch(
                        buf_b, buf_t, total_positions)
                    buf_b, buf_t = [], []
            if buf_b:
                total_positions = train_batch(buf_b, buf_t, total_positions)

        rms = float(np.sqrt(np.mean(losses[-500:]))) if losses else 0.0
        print(f"[train] Erreur RMS récente : {rms:.4f} | "
              f"{total_positions} positions en {(time.time()-t_start)/60:.1f} min")
    finally:
        stockfish.quit()

    # Servir le MEILLEUR checkpoint de validation s'il bat le hasard
    if best_val > 0.05 and Path(best_path).exists():
        shutil.copy(best_path, args.output)
        print(f"[train] Meilleur checkpoint retenu (val ρ = {best_val:+.3f}) → {args.output}")
    else:
        torch.save(readout.state_dict(), args.output)
        print(f"[train] Aucun checkpoint ne généralise (val ρ = {best_val:+.3f}) — "
              f"état final sauvegardé dans {args.output}")

    print(f"[train] Readout sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
