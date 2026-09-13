"""
FlyIntel benchmark CLI — score the fly's brain across domains and emit a
JSON leaderboard.

  python benchmark_fly.py                    # MaleCNS (reel) si present, sinon synth
  python benchmark_fly.py --synthetic        # force connectome synthetique
  python benchmark_fly.py --tag custom       # etiquette l'entree leaderboard
  python benchmark_fly.py --list             # affiche le leaderboard existant

Le leaderboard est ecrit dans benchmarks/leaderboard.json et merge par tag
(un nouveau run meme tag remplace l'ancien).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flyintel  # noqa: E402
from flyintel import bench  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true",
                   help="force le connectome synthetique (demo)")
    p.add_argument("--tag", default="run", help="etiquette leaderboard")
    p.add_argument("--list", action="store_true", help="affiche le leaderboard")
    args = p.parse_args()

    if args.list:
        if bench.LEADERBOARD_PATH.exists():
            lb = json.loads(bench.LEADERBOARD_PATH.read_text())
            print(bench.summary(lb))
        else:
            print("pas de leaderboard (lancer un benchmark d'abord)")
        return

    print(f"[flyintel] device={flyintel.default_device()} malecns={flyintel.has_malecns()}")
    brain, conn = flyintel.load_brain(force_synthetic=args.synthetic)
    n = int(conn["W"].shape[0])
    print(f"[flyintel] brain {n} neurones, {int(conn['W'].nnz)} synapses")

    # Readout eventuellement present (readout_looped.pt / readout.pt)
    readout = encoder = val_boards = targets = None
    for path in ("readout_looped.pt", "readout.pt",
                 "backend/readout_looped.pt", "backend/readout.pt"):
        if Path(path).exists():
            try:
                readout, meta = flyintel.load_readout_module(path, brain, conn)
                from flyintel import BoardEncoder
                encoder = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
                print(f"[flyintel] readout charge: {path}")
                break
            except Exception as e:
                print(f"[flyintel] readout {path} injouable ({e})")
    if readout is None:
        print("[flyintel] aucun readout entraine -> chess_reflex sera None")
    else:
        # Construit un jeu de validation d'echecs + cibles stockfish (reuse
        # make_val_set + stockfish depuis train_readout_looped).
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
            from train_readout import make_val_set
            import chess
            import chess.engine
            import time as _t
            stockfish = None
            for sf in ("tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe",
                       "tools/stockfish/stockfish-windows-x86-64-avx2.exe",
                       "tools/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe",
                       "stockfish"):
                if Path(sf).exists() or sf == "stockfish":
                    try:
                        stockfish = chess.engine.SimpleEngine.popen_uci(sf)
                        break
                    except Exception:
                        stockfish = None
            if stockfish is not None:
                val_boards = make_val_set(15, seed=77)
                targets = []
                for b in val_boards:
                    try:
                        info = stockfish.analyse(b, chess.engine.Limit(depth=6))
                        targets.append(info["score"].white().score(mate_score=10000) / 1000.0)
                    except Exception:
                        targets.append(0.0)
                stockfish.quit()
                print(f"[flyintel] {len(val_boards)} positions de validation + cibles stockfish")
        except Exception as e:
            print(f"[flyintel] pas de val-set echecs ({e})")

    print(f"[flyintel] lancement du benchmark ({args.tag})...")
    results = bench.run_all(brain, conn, readout=readout, encoder=encoder,
                            val_boards=val_boards, targets=targets, tag=args.tag)
    print(bench.summary([results]))
    print(f"[flyintel] leaderboard -> {bench.LEADERBOARD_PATH}")


if __name__ == "__main__":
    main()