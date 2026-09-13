"""Ablation honnête (témoin négatif permanent, rapport risque 2) :
readout linéaire entraîné (ancien régime) vs linéaire aléatoire vs looped pilote,
même val set (n=25, seed 77), même régime prod alpha/100 pas.
Usage : python scripts/ablate_readouts.py
"""
import os
import sys
from pathlib import Path

import argparse

import chess
import chess.engine
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from data_loader import load_connectome
from brain import build_prod_brain
from encoding import BoardEncoder
from readout import LinearReadout
from readout_looped import load_looped, trajectory_from_brain
from train_readout import make_val_set

SF = "tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="all",
                    choices=["all", "linear", "looped"])
    only = ap.parse_args().only
    conn = load_connectome()
    brain = build_prod_brain(conn)
    enc = BoardEncoder(int(conn["is_sensory"].sum()))
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        val = []
        for b in make_val_set(25, seed=77):
            info = sf.analyse(b, chess.engine.Limit(depth=8))
            val.append((b, info["score"].white().score(mate_score=10000) / 1000.0))
    finally:
        sf.quit()
    print(f"[ablate] {len(val)} positions cibles Stockfish", flush=True)

    nm, nd = int(conn["is_motor"].sum()), int(conn["is_descending"].sum())

    def rho_of(fn, label):
        preds = [fn(b) for b, _ in val]
        tg = [t for _, t in val]
        r = None if len(set(np.round(preds, 6))) < 3 else float(spearmanr(tg, preds).correlation)
        print(f"[ablate] {label} : rho = {r:+.3f}" if r is not None else f"[ablate] {label} : N/A (constant)", flush=True)

    def run_lin(rd):
        def f(b):
            with torch.no_grad():
                r = brain.run(enc.encode(b).to(brain.device), n_steps=100, record_every=10)
                return rd(r["motor_mean"], r["descending_last"]).item()
        return f

    st = torch.load("backend/readout.pt", map_location="cpu")
    lin_trained = LinearReadout(nm, nd)
    lin_trained.load_state_dict(st, strict=False)
    lin_trained.eval()
    if only in ("all", "linear"):
        rho_of(run_lin(lin_trained), "linear-entraine-ancien-regime")

        torch.manual_seed(1)
        rho_of(run_lin(LinearReadout(nm, nd).eval()), "linear-aleatoire")

    if only in ("all", "looped"):
        dec, _ = load_looped(os.path.join(os.environ["TEMP"], "looped_pilot.pt"), "cpu")

        def f_loop(b):
            with torch.no_grad():
                r = brain.run(enc.encode(b).to(brain.device), n_steps=100, record_every=10,
                              return_history=True)
                tr = trajectory_from_brain(r)
                dec.observe(tr.reshape(-1, tr.shape[-1]))
                return dec(tr)[-1].item()

        rho_of(f_loop, "looped-pilote-90pos")


if __name__ == "__main__":
    main()
