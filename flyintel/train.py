"""
flyintel training API — train the readout decoder on top of the frozen
connectome. Thin wrapper over backend/train_readout_looped.py (no duplicate
logic; ponytail: reuse). Exposes a programmatic API + a CLI.

  from flyintel import train
  result = train.train_looped(episodes=200, steps=50, output="models/rl.pt")
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

_BACKEND_TRAIN = Path(__file__).resolve().parent.parent / "backend" / "train_readout_looped.py"


def train_looped(
    episodes: int = 100,
    max_moves: int = 60,
    output: str = "readout_looped.pt",
    steps: int = 200,
    d_h: int = 128,
    window: int = 16,
    encoder: str = "classic",
    dn: str = "all",
    dn_k: int = 256,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 0,
    dry_run: bool = False,
    stockfish: str = "stockfish",
    depth: int = 8,
    val_n: int = 25,
    val_every: int = 20,
) -> dict:
    """Entraîne un LoopedReadout sur le connectome gelé. Retourne les métriques
    (best val rho) imprimées par le script sous-jacent. dry_run=True → fumée
    rapide sur connectome synthétique, sans Stockfish."""
    argv = [
        sys.argv[0],
        "--episodes", str(episodes),
        "--max-moves", str(max_moves),
        "--output", output,
        "--steps", str(steps),
        "--d-h", str(d_h),
        "--window", str(window),
        "--encoder", encoder,
        "--dn", dn,
        "--dn-k", str(dn_k),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--stockfish", stockfish,
        "--depth", str(depth),
        "--val-n", str(val_n),
        "--val-every", str(val_every),
    ]
    if dry_run:
        argv.append("--dry-run")
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(_BACKEND_TRAIN), run_name="__main__")
    finally:
        sys.argv = old
    # Le script écrit son propre log console; on retourne le chemin de sortie.
    return {"output": output, "dry_run": dry_run}


def main():
    p = argparse.ArgumentParser(description="Train the fly's readout decoder")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-moves", type=int, default=60)
    p.add_argument("--output", default="readout_looped.pt")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--d-h", type=int, default=128)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--encoder", default="classic", choices=["classic", "modal"])
    p.add_argument("--dn", default="all", choices=["all", "maxflow"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    train_looped(**vars(args))


if __name__ == "__main__":
    main()