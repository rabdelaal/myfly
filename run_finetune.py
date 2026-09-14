"""Lanceur du fine-tune complet LoopedReadout sur MaleCNS (mode détaché)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyintel import train

SF = "tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe"

if __name__ == "__main__":
    # Enregistre son propre PID pour le watcher (processus détaché).
    Path("finetune.pid").write_text(str(os.getpid()))
    train.train_looped(
        episodes=200,
        max_moves=60,
        steps=50,          # 50 pas suffisent pour la dynamique (RAPPORT B2)
        batch_size=8,
        val_n=25,
        val_every=20,
        output="readout_looped.pt",
        stockfish=SF,
        seed=42,
    )
    print("[finetune] TERMINE")