"""
Entraîne le LoopedReadout (RAPPORT B2) : full BPTT par autograd sur la
trajectoire, connectome GELÉ (no_grad côté brain.run — on ne traverse pas la
simulation, c'est le décodeur seul qui apprend).

  loss = MSE(score_T, stockfish) + 0.1 * MSE(moyenne_scores, stockfish)

Usage :
  python train_readout_looped.py --episodes 50 --dry-run   # synthétique, sans Stockfish
  python train_readout_looped.py --episodes 200 --steps 50 # vrai MaleCNS

Critère de succès : val rho > 0,3 (n≥25, 3 seeds) vs baseline linéaire 0,14.
Critère d'abandon : rho < 0,2 avec trajectoire + canaux modaux -> le goulot est
la simulation LIF, pas le décodeur (cf. rapport B2/risque 1).
"""
import argparse
import shutil
import time
from pathlib import Path

import chess
import numpy as np
import torch
from tqdm import tqdm

from data_loader import load_connectome, _synthetic_connectome, dn_maxflow_mask
from brain import FlyBrain
from encoding import BoardEncoder, ModalBoardEncoder
from readout_looped import LoopedReadout, trajectory_from_brain, save_looped
from train_readout import make_val_set


def _stockfish_or_none(path: str):
    import chess.engine
    for _ in range(5):
        try:
            return chess.engine.SimpleEngine.popen_uci(path)
        except Exception:
            time.sleep(5)
    return None


def validate_looped(dec, val_set, brain, encoder, steps, dn_idx=None) -> float | None:
    """rho de Spearman, validation BATCHÉE en un seul run (B=n)."""
    from scipy.stats import spearmanr
    boards = [b for b, _ in val_set]
    targets = [t for _, t in val_set]
    with torch.no_grad():
        currents = encoder.encode_batch(boards).to(brain.device)
        r = brain.run(currents, n_steps=steps, record_every=10, return_history=True)
        traj = trajectory_from_brain(r)
        if dn_idx is not None:
            motor_n = r["motor_hist"].shape[-1]
            keep = torch.tensor(np.r_[np.ones(motor_n, bool), dn_idx], device=traj.device)
            traj = traj[..., keep]
        dec.observe(traj.reshape(-1, traj.shape[-1]))
        preds = dec(traj)[-1].cpu().tolist()
    if len(set(np.round(preds, 6))) < 3:
        return None
    return float(spearmanr(targets, preds).correlation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-moves", type=int, default=60)
    p.add_argument("--output", type=str, default="readout_looped.pt")
    p.add_argument("--stockfish", type=str, default="stockfish")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--d-h", type=int, default=128)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--encoder", type=str, default="classic", choices=["classic", "modal"])
    p.add_argument("--dn", type=str, default="all", choices=["all", "maxflow"])
    p.add_argument("--dn-k", type=int, default=256)
    p.add_argument("--dry-run", action="store_true",
                   help="connectome synthetique + cibles aleatoires (fumee, sans Stockfish)")
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--val-n", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8,
                   help="positions par run connectome (B) : ~4x plus vite à 8")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dry_run:
        conn = _synthetic_connectome(n_neurons=1500, sparsity=0.008)
        conn["is_hotspot"] = np.zeros(len(conn["neuron_ids"]), bool)
        brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
        reward_of = lambda b: float(np.random.default_rng(args.seed).normal())  # noqa: E731
        val_set = [(b, float(np.random.default_rng(1).normal())) for b in make_val_set(8)]
        stockfish = None
    else:
        print("[looped-train] Chargement du connectome...")
        conn = load_connectome()
        from brain import build_prod_brain
        brain = build_prod_brain(conn)  # régime prod alpha (current = silence sur MaleCNS)
        stockfish = _stockfish_or_none(args.stockfish)
        if stockfish is None:
            raise SystemExit("[looped-train] Stockfish introuvable")

        def reward_of(board: chess.Board) -> float:
            import chess.engine
            try:
                info = stockfish.analyse(board, chess.engine.Limit(depth=args.depth))
                return info["score"].white().score(mate_score=10000) / 1000.0
            except Exception:
                return 0.0
        val_set = [(b, reward_of(b)) for b in make_val_set(args.val_n, seed=77)]
        print(f"[looped-train] {len(val_set)} positions de validation (n={args.val_n})")

    Enc = ModalBoardEncoder if args.encoder == "modal" else BoardEncoder
    encoder = Enc(n_sensory=int(conn["is_sensory"].sum()))

    # Sélection DN (A3) : masque booléen sur les descendants, ou None
    dn_idx = None
    if args.dn == "maxflow":
        mask = dn_maxflow_mask(conn, k=args.dn_k)
        desc_global = np.nonzero(np.asarray(conn["is_descending"]))[0]
        dn_idx = np.isin(desc_global, np.nonzero(mask)[0])
        print(f"[looped-train] DN max-flow : {int(dn_idx.sum())}/{len(desc_global)}")

    n_motor = int(conn["is_motor"].sum())
    n_desc = int(dn_idx.sum()) if dn_idx is not None else int(conn["is_descending"].sum())
    dec = LoopedReadout(n_motor + n_desc, d_h=args.d_h, window=args.window).to(brain.device)
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr)

    best_val, best_path, total, losses = -1.0, args.output + ".best", 0, []

    def _save_best(vr):
        nonlocal best_val
        best_val = vr
        save_looped(best_path, dec,
                    dn_mask=dn_idx if dn_idx is not None else None,
                    encoder=args.encoder)
        tqdm.write(f"  {total} pos | val rho = {vr:+.3f} * meilleur")

    def train_batch(boards_b, targets_b):
        """Un mini-batch : 1 seul run connectome pour K positions (B=K)."""
        nonlocal total
        prev = total
        tb = torch.tensor(targets_b, device=brain.device)
        with torch.no_grad():  # connectome gelé
            currents = encoder.encode_batch(boards_b).to(brain.device)
            r = brain.run(currents, n_steps=args.steps, record_every=10,
                          return_history=True)
            traj = trajectory_from_brain(r).detach()
            if dn_idx is not None:
                motor_n = r["motor_hist"].shape[-1]
                keep = torch.tensor(np.r_[np.ones(motor_n, bool), dn_idx],
                                    device=traj.device)
                traj = traj[..., keep]
            dec.observe(traj.reshape(-1, traj.shape[-1]))
        out = dec(traj)  # (T,K) — même objet qu'en inférence
        loss = torch.nn.functional.mse_loss(out[-1], tb) \
            + 0.1 * torch.nn.functional.mse_loss(out.mean(0), tb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        total += len(boards_b)
        if prev // args.val_every != total // args.val_every:
            vr = validate_looped(dec, val_set, brain, encoder, args.steps, dn_idx)
            if vr is not None and vr > best_val:
                _save_best(vr)
        if prev // 10 != total // 10 and losses:
            tqdm.write(f"  {total} pos | loss {np.mean(losses[-50:]):.3f} | "
                       f"{(time.time()-t0)/total:.1f} s/pos")

    t0 = time.time()
    try:
        for _ in tqdm(range(args.episodes), desc="Épisodes"):
            board = chess.Board()
            buf_b, buf_t = [], []
            for _ in range(args.max_moves):
                if board.is_game_over():
                    break
                board.push(np.random.choice(list(board.legal_moves)))
                buf_b.append(board.copy())
                buf_t.append(reward_of(board))
                if len(buf_b) >= args.batch_size:
                    train_batch(buf_b, buf_t)
                    buf_b, buf_t = [], []
            if buf_b:
                train_batch(buf_b, buf_t)
    finally:
        if stockfish is not None:
            stockfish.quit()

    print(f"[looped-train] {total} positions en {(time.time()-t0)/60:.1f} min | "
          f"meilleur val rho = {best_val:+.3f} (succès si > +0,30, abandon si < +0,20)")
    if best_val > 0.05 and Path(best_path).exists():
        shutil.copy(best_path, args.output)
        print(f"[looped-train] checkpoint retenu -> {args.output}")
    else:
        save_looped(args.output, dec, dn_mask=dn_idx, encoder=args.encoder)
        print(f"[looped-train] pas de généralisation — état final -> {args.output}")


if __name__ == "__main__":
    main()
