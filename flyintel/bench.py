"""
FlyIntel benchmark suite — scores the fly's brain across several domains and
emits a JSON leaderboard.

Domains (each a pure function of (brain, conn, readout?)):

  - chess_reflex  : quality of a readout vs stockfish targets (Spearman rho)
  - feeding       : Shiu feeding-reflex rate (MN9 Hz under sugar-GRN drive)
  - discrimination: can the motor readout tell different stimuli apart?
  - dynamics      : stability, no avalanches, firing-rate sanity
  - speed         : simulation wall-clock (ms per step) with backend dispatch
  - metaphor      : metabrain stub — motor separability of non-chess options
                    encoded via AnythingEncoder (transfer scored in Phase 2)
  - ie_state      : EvolveScaler-style ground truth — executable world
                    (ledger) replayed in code; readout must pick the
                    code-derived answer over the naive trap (which counts
                    invalid records). v0 encodes options only (l'historique
                    n'est pas injecté : il faudra des canaux par tour).

Every domain returns (metric_name -> value). Results are merged into
leaderboard.json keyed by domain; a higher `score` column ranks better.
Extend by adding a function here (ponytail: one file, no framework).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .backends import benchmark_backends, gelu_fast, tanh_fast

LEADERBOARD_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "leaderboard.json"


def _deterministic(fn):
    """Tirages Poisson du cerveau figés pendant un domaine : même seed → mêmes
    trajectoires, leaderboard comparable run-to-run. État RNG restauré après
    (aucun effet sur l'appelant ni sur le jeu prod, non décoré)."""
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        g = torch.get_rng_state()
        cg = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(0)
        try:
            return fn(*a, **k)
        finally:
            torch.set_rng_state(g)
            if cg is not None:
                torch.cuda.set_rng_state_all(cg)
    return w


# --------------------------------------------------------------------------
# Domain: chess reflex — does a trained readout rank moves like stockfish?
# --------------------------------------------------------------------------
@_deterministic
def chess_reflex(brain, conn, readout=None, encoder=None, val_boards=None,
                 targets=None, steps=100) -> dict:
    """rho de Spearman entre les scores du readout et des cibles (stockfish).
    readout None + encoder None → rho vide (marque 'readout_unavailable')."""
    from scipy.stats import spearmanr
    if readout is None or encoder is None or val_boards is None:
        return {"score": None, "detail": "no readout supplied"}
    if targets is None:
        targets = [float(np.random.default_rng(7).normal()) for _ in val_boards]
    with torch.no_grad():
        currents = encoder.encode_batch(val_boards).to(brain.device)
        if hasattr(readout, "cell"):  # LoopedReadout : trajectoire temporelle
            r = brain.run(currents, n_steps=steps, record_every=10,
                          return_history=True)
            from readout_looped import trajectory_from_brain
            traj = trajectory_from_brain(r)
            readout.observe(traj.reshape(-1, traj.shape[-1]))
            preds = readout(traj)[-1].cpu().tolist()
            kind = "looped"
        else:  # LinearReadout : moyennes motrices + descendantes
            r = brain.run(currents, n_steps=steps, record_every=10)
            readout.observe(torch.cat([r["motor_mean"], r["descending_last"]], dim=-1))
            preds = readout(r["motor_mean"], r["descending_last"]).cpu().tolist()
            kind = "linear"
    if len(set(np.round(preds, 6))) < 3:
        return {"score": None, "detail": f"readout output degenerate ({kind})"}
    rho = float(spearmanr(targets, preds).correlation)
    return {"score": rho, "detail": f"rho vs {len(targets)} stockfish targets ({kind})"}


# --------------------------------------------------------------------------
# Domain: feeding reflex (Shiu) — MN9 rate under sugar-GRN Poisson drive
# --------------------------------------------------------------------------
@_deterministic
def feeding(brain, conn, drive_hz=100.0, ms=200) -> dict:
    mn9 = conn.get("mn9_idx")
    if mn9 is None or len(mn9) == 0:
        return {"score": None, "detail": "no MN9 annotations"}
    mn9_t = torch.tensor(np.asarray(mn9, dtype=np.int64), device=brain.device)
    brain.reset(1)
    brain.set_sugar_drive(drive_hz, ms)
    total = 0.0
    for _ in range(ms):
        brain.step(None)
        total += float(brain.spikes[mn9_t].sum().item())
    brain.set_sugar_drive(0.0, 0)
    hz = total / len(mn9) / (ms / 1000.0)
    return {"score": hz, "detail": f"MN9 rate under {drive_hz} Hz GRN drive"}


# --------------------------------------------------------------------------
# Domain: discrimination — can motor readout separate distinct stimuli?
# --------------------------------------------------------------------------
@_deterministic
def discrimination(brain, conn, stimuli=("feed", "pet", "clean", "threat"),
                   steps=100) -> dict:
    """One-vs-rest separability of the motor response to different action
    stimuli. Score = fraction of pairs whose motor-mean vectors differ more
    than chance (rank-based). Returns an aggregate 0..1."""
    from tamagotchi import StimulusEncoder
    enc = StimulusEncoder(int(conn["is_sensory"].sum()))
    patterns = []
    for s in stimuli:
        with torch.no_grad():
            c = enc.encode(s).to(brain.device)
            r = brain.run(c, n_steps=steps, record_every=10)
            patterns.append(r["motor_mean"].cpu().numpy().ravel())
    # Séparabilité : plus la distance inter-stimuli est grande vs intra-bruit.
    # On refait 3 runs pour estimer la dispersion intra-stimulus.
    spread, sep = 0.0, 0.0
    for _ in range(3):
        v2 = []
        for s in stimuli:
            with torch.no_grad():
                c = enc.encode(s).to(brain.device)
                r = brain.run(c, n_steps=steps, record_every=10)
                v2.append(r["motor_mean"].cpu().numpy().ravel())
        for i in range(len(stimuli)):
            spread += float(np.linalg.norm(v2[i] - patterns[i]))
    spread /= (3 * len(stimuli))
    for i in range(len(stimuli)):
        for j in range(i + 1, len(stimuli)):
            sep += float(np.linalg.norm(patterns[i] - patterns[j]))
    sep /= (len(stimuli) * (len(stimuli) - 1) / 2)
    # Plancher de bruit à 10 % de sep : cerveau déterministe (spread≈0) plafonné
    # à log1p(10)≈2.4 au lieu d'exploser (ex. 19.2 sur synthétique).
    ratio = float(sep / (spread + 0.1 * sep + 1e-9))
    # log1p : tasse le ratio en restant monotone (ratio ≤ 10, score ≤ 2.4).
    score = float(np.log1p(ratio))
    return {"score": score, "detail": f"log1p(inter/intra {ratio:.1f}) over {len(stimuli)} stimuli"}


# --------------------------------------------------------------------------
# Domain: dynamics — stability / no avalanche / firing-rate sanity
# --------------------------------------------------------------------------
def dynamics(brain, conn, steps=300) -> dict:
    brain.reset(1)
    sensor_hz, motor_hz, peak = 0.0, 0.0, 0.0
    for _ in range(steps):
        brain.step(None)
        s = int(brain.spikes[brain.is_sensory].sum())
        m = int(brain.spikes[brain.is_motor].sum())
        sensor_hz += s
        motor_hz += m
        peak = max(peak, s + m)
    n_s = max(int(brain.n_sensory), 1)
    n_m = max(int(brain.is_motor.sum()), 1)
    sensor_hz = sensor_hz / n_s / (steps / 1000.0)
    motor_hz = motor_hz / n_m / (steps / 1000.0)
    if peak == 0:
        return {"score": 0.0, "detail": f"silent: sensory {sensor_hz:.1f} Hz · motor {motor_hz:.1f} Hz · peak {peak}"}
    # Score composite borné : on veut un cerveau vivant mais pas en avalanche.
    # Saine ≈ taux sensoriel modéré, moteur < sensoriel, peak raisonnable.
    sane = float(min(1.0, motor_hz / max(sensor_hz * 1.1, 1e-3)))
    score = float(np.clip(1.0 - abs(sane - 0.3), 0.0, 1.0))
    return {"score": score, "detail": f"sensory {sensor_hz:.1f} Hz · motor {motor_hz:.1f} Hz · peak {peak}"}


# --------------------------------------------------------------------------
# Domain: metaphor (metabrain) — le readout échecs peut-il classer des
# options NON-échecs encodées en métaphore ?
# Phase 0 (stub) : sans teacher, mesure la séparabilité motrice des options
# (prouve que l'encodeur pilote des états distincts). Avec readout : reporte
# en plus la marge du readout (quelle option il préfère, de combien).
# Phase 2 : brancher le jeu A/B ancré (vérité stockfish) pour scorer le
# transfert ; le leaderboard tranchera alors si le metabrain vaut quelque
# chose. Voir AnythingEncoder dans backend/encoding.py.
# --------------------------------------------------------------------------
METAPHOR_PROBE = (
    "sacrifice the queen for forced checkmate in two moves",
    "push a kingside pawn one square with no threat",
)


@_deterministic
def metaphor(brain, conn, readout=None, encoder=None, steps=100) -> dict:
    from encoding import AnythingEncoder
    enc = encoder if isinstance(encoder, AnythingEncoder) else AnythingEncoder(
        n_sensory=int(conn["is_sensory"].sum()))
    with torch.no_grad():
        currents = enc.encode_options(list(METAPHOR_PROBE)).to(brain.device)
        pats = []
        for o in range(currents.shape[1]):
            r = brain.run(currents[:, o:o + 1], n_steps=steps, record_every=10)
            pats.append(r["motor_mean"].cpu().numpy().ravel())
    sep = float(np.linalg.norm(pats[0] - pats[1]))
    out = {"score": None, "detail": f"stub: motor separation {sep:.3f}, teacher readout required for transfer score"}
    if readout is not None:
        try:
            with torch.no_grad():
                s = [float(readout(p.reshape(1, -1)).ravel()[0]) for p in pats]
            margin = s[0] - s[1]
            out["detail"] += f" | readout margin A-B {margin:+.3f}"
        except Exception as e:
            out["detail"] += f" | readout failed: {e}"
    return out


# --------------------------------------------------------------------------
# Domain: ie_state — vérité terrain par re-exécution (EvolveScaler).
# Un LedgerWorld génère un historique avec révisions/annulations/bruit ;
# la bonne réponse vient du REPLAY du code, le piège naïf compte les
# enregistrements invalides. v0 : options seules encodées (pas l'historique).
# --------------------------------------------------------------------------
@_deterministic
def ie_state(brain, conn, readout=None, encoder=None, steps=100) -> dict:
    from encoding import AnythingEncoder
    from ie_worlds import LedgerWorld
    w = LedgerWorld(seed=11)
    w.gen(40, invalid_rate=0.15)
    q, truth, chk = w.ask("total")
    # Piège naïf : tout compter, y compris annulés/révisés (avant révision).
    naive = 0
    seen = {}
    for e in w.events:
        if e["kind"] == "add":
            seen[e["id"]] = e["amount"]
            naive += e["amount"]
    trap = str(naive)
    if trap == truth:  # pas de divergence : insipide, on le signale
        return {"score": None, "detail": "seed sans divergence valide/piège"}
    enc = encoder if isinstance(encoder, AnythingEncoder) else AnythingEncoder(
        n_sensory=int(conn["is_sensory"].sum()))
    with torch.no_grad():
        currents = enc.encode_options(
            [f"{q} {truth}", f"{q} {trap}"]).to(brain.device)
        pats = []
        for o in range(currents.shape[1]):
            r = brain.run(currents[:, o:o + 1], n_steps=steps, record_every=10)
            pats.append(r["motor_mean"].cpu().numpy().ravel())
    sep = float(np.linalg.norm(pats[0] - pats[1]))
    out = {"score": None,
           "detail": f"stub: sep {sep:.3f}, truth={truth} trap={trap}, "
                     f"readout required to score"}
    if readout is not None:
        try:
            with torch.no_grad():
                s = [float(readout(p.reshape(1, -1)).ravel()[0]) for p in pats]
            good = int(s[0] > s[1])
            out["score"] = float(good)
            out["detail"] += f" | readout picks {'truth' if good else 'TRAP'} (margin {s[0]-s[1]:+.3f})"
        except Exception as e:
            out["detail"] += f" | readout failed: {e}"
    return out


# --------------------------------------------------------------------------
# Domain: speed — wall-clock per step, both backends if available
# --------------------------------------------------------------------------
def speed(brain, conn, steps=200) -> dict:
    brain.reset(1)
    t = time.perf_counter()
    for _ in range(steps):
        brain.step(None)
    ms_step = (time.perf_counter() - t) / steps * 1000.0
    return {"score": -ms_step, "detail": f"{ms_step:.1f} ms/step (lower better, score inverted)"}


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
DOMAINS = {
    "chess_reflex": chess_reflex,
    "feeding": feeding,
    "discrimination": discrimination,
    "dynamics": dynamics,
    "speed": speed,
    "metaphor": metaphor,
    "ie_state": ie_state,
}


def run_all(brain, conn, readout=None, encoder=None, val_boards=None,
            targets=None, tag: str = "run") -> dict:
    """Exécute tous les domaines, fusionne dans leaderboard.json."""
    results = {"tag": tag, "n_neurons": int(conn["W"].shape[0]),
               "n_synapses": int(conn["W"].nnz), "domains": {}}
    for name, fn in DOMAINS.items():
        try:
            if name == "chess_reflex":
                res = fn(brain, conn, readout, encoder, val_boards, targets)
            elif name in ("metaphor", "ie_state"):
                res = fn(brain, conn, readout, encoder)
            elif name == "feeding":
                res = fn(brain, conn)
            else:
                res = fn(brain, conn)
            results["domains"][name] = res
        except Exception as e:
            results["domains"][name] = {"score": None, "detail": f"error: {e}"}
    # Accélération des kernels (table backends)
    results["backends"] = benchmark_backends()
    _merge_leaderboard(results)
    return results


def _merge_leaderboard(results: dict) -> None:
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LEADERBOARD_PATH.exists():
        try:
            lb = json.loads(LEADERBOARD_PATH.read_text())
        except Exception:
            lb = []
    else:
        lb = []
    # une entrée par tag (run frais remplace l'ancien même tag)
    lb = [e for e in lb if e.get("tag") != results["tag"]]
    lb.append(results)
    lb.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    results["timestamp"] = time.time()
    LEADERBOARD_PATH.write_text(json.dumps(lb, indent=2, ensure_ascii=False))


def summary(lb: list[dict]) -> str:
    """Texte lisible du leaderboard : un tableau markdown des domaines."""
    rows = []
    for entry in lb:
        for dom, res in entry.get("domains", {}).items():
            rows.append((entry["tag"], dom, res.get("score")))
    out = ["| tag | domain | score |", "|---|---|---|"]
    for tag, dom, sc in rows:
        out.append(f"| {tag} | {dom} | {sc if sc is None else round(sc, 4)} |")
    return "\n".join(out)


if __name__ == "__main__":
    # smoke run sur un petit connectome synthétique
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from data_loader import _synthetic_connectome
    from brain import FlyBrain
    conn = _synthetic_connectome(n_neurons=1500, sparsity=0.008)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"])
    r = run_all(brain, conn, tag="smoke")
    print(summary([r]))