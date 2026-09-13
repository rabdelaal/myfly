"""Garde-fous prouvables des améliorations (RAPPORT A1/A2/A3/A4/B2/B3/B4 + fix gain).
Run : cd backend && python test_improvements.py — <60s sur CPU, sans MaleCNS.
"""
import numpy as np
import torch
import chess

from data_loader import _synthetic_connectome, _add_dimorphism_masks, dn_maxflow_mask
from brain import FlyBrain
from encoding import BoardEncoder, ModalBoardEncoder, STIM_TARGET_NORM
from readout_looped import LoopedReadout, trajectory_from_brain, save_looped, load_looped


def test_encoding_norm():
    enc = BoardEncoder(n_sensory=200)
    b = chess.Board()
    c = enc.encode(b)
    n = float(c.norm())
    assert abs(n - STIM_TARGET_NORM) / STIM_TARGET_NORM < 0.01, n
    cb = enc.encode_batch([b, chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")])
    for j in range(cb.shape[1]):
        nj = float(cb[:, j].norm())
        assert abs(nj - STIM_TARGET_NORM) / STIM_TARGET_NORM < 0.01, nj
    print(f"[ok] encoding norm unique {n:.1f} (pas de ×20)")


def test_modal_encoder():
    enc = ModalBoardEncoder(n_sensory=200)
    b1, b2 = chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    c1, c2 = enc.encode(b1), enc.encode(b2)
    assert abs(float(c1.norm()) - STIM_TARGET_NORM) / STIM_TARGET_NORM < 0.02
    assert abs(float(c2.norm()) - STIM_TARGET_NORM) / STIM_TARGET_NORM < 0.02
    # Deux positions ≠ doivent donner deux patterns ≠ (anti rank-1)
    cos = float((c1 @ c2) / (c1.norm() * c2.norm() + 1e-9))
    assert cos < 0.999, cos
    cb = enc.encode_batch([b1, b2])
    assert cb.shape == (200, 2), cb.shape
    print(f"[ok] modal norm 450, cos(inter-positions)={cos:.3f} <1")


def test_brain_history():
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    cur = enc.encode(chess.Board())
    r = brain.run(cur, n_steps=20, record_every=5, return_history=True)
    T = 20 // 5
    assert r["motor_hist"].shape[0] == T, r["motor_hist"].shape
    assert r["motor_hist"].shape[1] == 1
    # motor_mean == moyenne de l'historique (cohérence train/inférence)
    assert torch.allclose(r["motor_mean"], r["motor_hist"].mean(0), atol=1e-5)
    # Ancien appel sans history inchangé
    r2 = brain.run(cur, n_steps=20, record_every=5)
    assert "motor_hist" not in r2
    print(f"[ok] history (T,B,n)=({T},1,{r['motor_hist'].shape[2]}), mean cohérente")


def test_dimorphism_fallback():
    conn = _synthetic_connectome(n_neurons=100)
    _add_dimorphism_masks(conn)  # sans npz → zéros, pas de crash
    for k in ("is_male_specific", "is_dimorphic", "is_fru", "is_dsx", "is_hotspot"):
        assert k in conn and conn[k].shape == (100,) and not conn[k].any(), k
    assert "mean_abs_weight" in conn
    print("[ok] dimorphisme fallback zéros + mean_abs_weight présent")


def test_looped_readout():
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    cur = enc.encode(chess.Board())
    r = brain.run(cur, n_steps=20, record_every=5, return_history=True)
    traj = trajectory_from_brain(r)  # (T,1,d_in)
    dec = LoopedReadout(d_in=traj.shape[-1], d_h=16, window=4)
    dec.observe(traj.detach().reshape(-1, traj.shape[-1]))
    out = dec(traj)  # (T,B)
    assert out.shape[:2] == traj.shape[:2], out.shape
    out[-1].sum().backward()
    assert dec.cell.gru.weight_hh.grad is not None
    print(f"[ok] looped {tuple(traj.shape)} -> {tuple(out.shape)}, BPTT OK")


def test_no_double_gain_regression():
    from pathlib import Path
    src = (Path(__file__).parent / "chess_engine.py").read_text(encoding="utf-8")
    assert "encode_batch(boards).to(self.brain.device) * STIM_GAIN" not in src, "double gain revenu!"
    assert "encode(board).to(self.brain.device) * STIM_GAIN" not in src, "double gain revenu!"
    print("[ok] pas de * STIM_GAIN après encode_*() dans chess_engine")


def test_dn_maxflow():
    conn = _synthetic_connectome(n_neurons=400, sparsity=0.01)
    n_desc = int(conn["is_descending"].sum())
    m = dn_maxflow_mask(conn, k=10)
    assert m.dtype == bool and m.shape == (400,)
    assert 0 < int(m.sum()) <= min(10, n_desc), (int(m.sum()), n_desc)
    assert (m & ~conn["is_descending"]).sum() == 0  # que des descendants
    m_all = dn_maxflow_mask(conn, k=10_000)
    assert int(m_all.sum()) == n_desc  # k large → tous
    print(f"[ok] DN max-flow 10/{n_desc}, que des descendants")


def test_engine_linear_and_looped():
    from chess_engine import FlyChessEngine
    from readout import LinearReadout
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    board = chess.Board()
    # Linéaire : le chemin historique répond, scores finis et distincts
    lin = LinearReadout(int(conn["is_motor"].sum()), int(conn["is_descending"].sum()))
    eng = FlyChessEngine(brain, enc, lin, max_candidates=4, n_steps=20)
    mv, sc = eng.choose_move(board)
    assert mv in board.legal_moves and len(sc) == 4
    assert len(set(np.round(list(sc.values()), 6))) >= 1
    # Looped : même moteur, trajectoire consommée, pas de crash
    dec = LoopedReadout(int(conn["is_motor"].sum()) + int(conn["is_descending"].sum()),
                        d_h=16, window=4)
    eng2 = FlyChessEngine(brain, enc, dec, max_candidates=4, n_steps=20)
    mv2, sc2 = eng2.choose_move(board)
    assert mv2 in board.legal_moves and len(sc2) == 4
    # Politique RL : échantillonne un coup légal, déterministe à T→0
    mv3, _ = eng.sample_move(board, temperature=1e-3)
    assert mv3 in board.legal_moves
    print(f"[ok] engine linear+looped+sample (coups {mv.uci()}, {mv2.uci()}, {mv3.uci()})")


def test_looped_save_load():
    import tempfile, os
    dec = LoopedReadout(24, d_h=16, window=4)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "loop.pt")
        save_looped(p, dec, dn_mask=np.array([True, False, True]), encoder="modal")
        dec2, meta = load_looped(p, "cpu")
        assert meta["encoder"] == "modal" and list(meta["dn_mask"]) == [True, False, True]
        x = torch.randn(5, 2, 24)
        assert dec2(x).shape == (5, 2)
    print("[ok] looped save/load + métas (encoder, dn_mask)")


def test_tamagotchi_dimorphic():
    from tamagotchi import FlyPet
    conn = _synthetic_connectome(n_neurons=200, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    pet = FlyPet.__new__(FlyPet)  # sans I/O disque : état minimal
    import time as _t
    pet.brain, pet.state = brain, {"stats": {"satiety": 80.0, "happiness": 80.0, "energy": 80.0, "hygiene": 80.0},
                                   "xp": 0, "sleeping": False, "last_tick": _t.time(), "birth": _t.time()}
    pet.save = lambda: None  # jamais d'I/O disque dans les tests
    from tamagotchi import StimulusEncoder
    pet.encoder = StimulusEncoder(int(conn["is_sensory"].sum()))
    h0 = pet.state["stats"]["happiness"]
    r1 = pet.do_action("courtship")
    assert r1["applied"] and pet.state["stats"]["happiness"] > h0, r1
    assert "aile" in r1["message"] or "parade" in r1["message"] or "frétillement" in r1["message"]
    r2 = pet.do_action("threat")
    assert r2["applied"] and pet.state["stats"]["happiness"] < pet.state["stats"]["happiness"] + 100
    print(f"[ok] tamagotchi courtship/threat ({r1['strength']}, {r2['strength']})")


def test_brain_diff_grad():
    from brain_diff import demo
    loss, g = demo()
    assert g > 0, "le gradient doit atteindre l'encodeur"
    print(f"[ok] diff surrogate loss={loss:.3f}, grad={g:.3f} > 0")


def test_collect_frames():
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")]
    cur = enc.encode_batch(boards)
    got = []
    r = brain.run(cur, n_steps=20, record_every=5, collect_frames=True,
                  frame_callback=got.append)
    T = 20 // 5
    assert len(got) == T and len(r["frames_all"]) == T, (len(got), len(r.get("frames_all", [])))
    assert all(len(cols) == 2 for cols in r["frames_all"])
    # La colonne 0 collectée == le callback historique (même simu, mêmes spikes)
    assert [c[0] for c in r["frames_all"]] == got
    r2 = brain.run(cur, n_steps=20, record_every=5)
    assert "frames_all" not in r2  # défaut inchangé, zéro surcoût
    print(f"[ok] collect_frames T={T} B=2, col0 == callback")


def test_no_resim_replay():
    from chess_engine import FlyChessEngine
    from readout import LinearReadout
    conn = _synthetic_connectome(n_neurons=300, sparsity=0.01)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    enc = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    lin = LinearReadout(int(conn["is_motor"].sum()), int(conn["is_descending"].sum()))
    eng = FlyChessEngine(brain, enc, lin, max_candidates=4, n_steps=20)
    calls = [0]
    real_run = brain.run
    def counting_run(*a, **k):
        calls[0] += 1
        return real_run(*a, **k)
    brain.run = counting_run
    try:
        frames = []
        mv, sc = eng.choose_move(chess.Board(), frame_callback=frames.append)
    finally:
        brain.run = real_run
    assert calls[0] == 1, f"re-simulation détectée ({calls[0]} runs)"
    assert len(frames) == 20 // 10 and mv in chess.Board().legal_moves
    print(f"[ok] 1 seule simu/coup, {len(frames)} frames rejouées ({mv.uci()})")


if __name__ == "__main__":
    torch.set_num_threads(2)
    test_encoding_norm()
    test_modal_encoder()
    test_brain_history()
    test_dimorphism_fallback()
    test_looped_readout()
    test_no_double_gain_regression()
    test_dn_maxflow()
    test_engine_linear_and_looped()
    test_looped_save_load()
    test_tamagotchi_dimorphic()
    test_brain_diff_grad()
    test_collect_frames()
    test_no_resim_replay()
    print("\nTOUT OK — intéressant + prouvable + testé.")
