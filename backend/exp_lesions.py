"""
LÉSIONS VIRTUELLES sur le connectome MaleCNS — quelle partie du cerveau fait quoi ?

Principe : silencier un groupe de neurones (leurs spikes sont forcés à zéro à
chaque pas — ils ne contribuent plus à rien, ni récurrence ni drive) et mesurer
le déficit comportemental sur TROIS assays :

  1. RÉFLEXE D'ALIMENTATION (Shiu et al., Nature 2024 — notre assay validé) :
     drive Poisson 100 Hz sur les GRN sucrées → taux de décharge des
     motoneurones MN9. Déficit = perte de Hz vs cerveau intact.
  2. SPONTANÉ : bruit membranaire σ = 0,3 mV sans drive — vitalité du réseau
     (taux global, et sensoriel/moteur séparés).
  3. DISCRIMINATION SENSORIMOTRICE : 4 sous-ensembles aléatoires de neurones
     sensoriels stimulés à 100 Hz → 4 patterns moteurs. Spécificité = 1 −
     corrélation moyenne entre patterns (le problème rank-1 du readout, mesuré
     par lésion).

Toutes les conditions sont simulées EN PARALLÈLE : chaque condition est une
colonne du batch (n_neurones, B) avec son propre masque de lésion — ×B plus
rapide qu'une condition par run.

Modèle LIF α exact de Shiu et al. (identique à exp_feeding_reflex.py) :
  V_rest = V_reset = −52 mV, V_th = −45 mV, τ_m = 20 ms, τ_syn = 5 ms,
  délai 1,8 ms (2 pas), réfractaire 2,2 ms (3 pas), Wsyn = 0,164 mV/synapse
  (notre calibration 80 % du max).

Usage : python exp_lesions.py [--quick]
Résultats : console + data/lesion_results.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

torch.set_num_threads(4)

DATA_DIR = Path(__file__).parent.parent / "data"
DT = 1.0
V_REST = V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0
TAU_SYN = 5.0
DELAY_STEPS = 2
REFRACTORY_STEPS = 3
WSYN = 0.164                 # mV/synapse (calibration 80 %, exp_feeding_reflex)
STIM_HZ = 100.0
REFLEX_MS = 800
SPONT_MS = 800
DISCR_MS = 600
DISCR_SETS = 4
NOISE_SIGMA = 0.3            # mV, bruit d'ambiance (cf. tamagotchi)

# --- Groupes de lésion (colonnes `class`/`superclass`/`dimorphism`/`fruDsx`
# des annotations MaleCNS v1.0 — celles du papier Cell 2026) ---
GROUPS = [
    ("intact (témoin)", None, None),
    ("corps du champignon (MB)", "class", "Kenyon_Cell"),
    ("complexe central (CX)", "class", "CX"),
    ("dimorphes mâle/femelle", "dimorphism", None),      # les 4 catégories
    ("fruitless/doublesex", "fruDsx", None),             # toute valeur annotée
    ("neurones descendants", "superclass", "descending_neuron"),
    ("lobes optiques", "superclass", "ol_intrinsic"),
    ("témoin aléatoire ~4k", "__random__", 4000),
    ("témoin aléatoire ~89k", "__random__", 89000),
]


def load_everything():
    """Connectome signé brut (npz) + annotations papier (feather) → groupes."""
    d = np.load(DATA_DIR / "malecns.npz")
    W = sp.csr_matrix((d["weights"], d["indices"], d["indptr"]),
                      shape=(len(d["neuron_ids"]), len(d["neuron_ids"])))
    neuron_ids = d["neuron_ids"]
    sorted_ids = np.sort(neuron_ids)
    order = np.argsort(neuron_ids)

    def idx_of(body_ids):
        pos = np.searchsorted(sorted_ids, body_ids)
        ok = (pos < len(sorted_ids)) & \
             (sorted_ids[np.clip(pos, 0, len(sorted_ids) - 1)] == body_ids)
        out = np.full(len(body_ids), -1, dtype=np.int64)
        out[ok] = order[pos[ok]]
        return out

    import pyarrow.feather as feather
    ann = feather.read_table(DATA_DIR / "body-annotations.feather",
                             memory_map=True)
    df = ann.select(["bodyId", "class", "superclass", "type", "dimorphism",
                     "fruDsx", "status"]).to_pandas()

    conn_idx = idx_of(df["bodyId"].to_numpy())
    df = df.assign(conn_idx=conn_idx)
    df = df[df["conn_idx"] >= 0]

    def group_mask(col, value=None):
        if col is None:
            return np.zeros(len(neuron_ids), dtype=bool)
        if col == "__random__":
            return None  # géré par l'appelant (taille dépend du tirage)
        if value is None:  # toute valeur annotée non nulle
            sel = df[col].notna() & (df[col].astype(str) != "")
        else:
            sel = df[col] == value
        m = np.zeros(len(neuron_ids), dtype=bool)
        m[df.loc[sel, "conn_idx"].to_numpy()] = True
        return m

    grn = df[df["type"].fillna("").str.contains("tpGRN", case=False)]
    grn_idx = idx_of(grn["bodyId"].to_numpy())
    grn_idx = grn_idx[grn_idx >= 0]
    has_out = np.asarray(W[grn_idx].sum(axis=1)).ravel() != 0
    grn_idx = grn_idx[has_out]

    motor_sc = ["vnc_motor", "cb_motor", "vnc_efferent"]
    motor_mask = np.zeros(len(neuron_ids), dtype=bool)
    motor_mask[df.loc[df["superclass"].isin(motor_sc), "conn_idx"].to_numpy()] = True

    sens_sc = ["ol_sensory", "cb_sensory", "vnc_sensory"]
    sens_mask = np.zeros(len(neuron_ids), dtype=bool)
    sens_mask[df.loc[df["superclass"].isin(sens_sc), "conn_idx"].to_numpy()] = True
    # neurones sensoriels avec au moins une synapse sortante (sinon drive stérile)
    has_out_all = np.asarray(W.sum(axis=1)).ravel() != 0
    sens_pool = np.where(sens_mask & has_out_all)[0]

    mn9 = df[df["type"].fillna("").str.contains("MN9", case=False)]
    mn9_idx = idx_of(mn9["bodyId"].to_numpy())
    mn9_idx = mn9_idx[mn9_idx >= 0]

    return W, neuron_ids, df, group_mask, grn_idx, mn9_idx, motor_mask, sens_pool


def build_conditions(rng, group_mask, n_neurons):
    """Liste (nom, masque_lesion (n,) bool) ; les aléatoires sont tirés pour
    matcher l'ordre de grandeur des groupes spécifiques (MB/CX ~4k, OL ~89k)."""
    conds = []
    sizes = {}
    for name, col, value in GROUPS:
        if col == "__random__":
            m = np.zeros(n_neurons, dtype=bool)
            m[rng.choice(n_neurons, size=int(value), replace=False)] = True
        elif col is None:
            m = np.zeros(n_neurons, dtype=bool)
        else:
            m = group_mask(col, value)
        conds.append((name, m))
        sizes[name] = int(m.sum())
    return conds, sizes


def simulate_phase(W_t, keep_mask, n_cond, ms, drive_fn=None, noise=False,
                   seed=0):
    """Un phase de simulation batchée. keep_mask : (n, B) float 0/1 (1 = vivant).
    drive_fn(step) -> (n, B) spikes de drive (déjà placés). Retourne
    spikes_sum (n, B) cumulés sur la phase."""
    n = W_t.shape[0]
    B = n_cond
    V = torch.full((n, B), V_REST)
    g = torch.zeros(n, B)
    refr = torch.zeros(n, B, dtype=torch.int32)
    delay_buf = [torch.zeros(n, B) for _ in range(DELAY_STEPS)]
    decay_syn = float(np.exp(-DT / TAU_SYN))
    spikes_sum = torch.zeros(n, B)
    spikes = torch.zeros(n, B)
    gen = torch.Generator().manual_seed(seed)
    steps = int(ms / DT)
    with torch.no_grad():
        for step in range(steps):
            delayed = delay_buf.pop(0)
            delay_buf.append(spikes.clone())
            inc = torch.sparse.mm(W_t, delayed)
            g = g * decay_syn + inc
            V = V + (-(V - V_REST) + g) * (DT / TAU_M)
            if noise:
                V = V + torch.randn(n, B, generator=gen) * NOISE_SIGMA
            spk = (V >= V_TH) & (refr <= 0)
            V = torch.where(spk, torch.full_like(V, V_RESET), V)
            refr = torch.where(spk,
                               torch.full_like(refr, REFRACTORY_STEPS,
                                               dtype=torch.int32),
                               (refr - 1).clamp(min=0))
            spikes = spk.float()
            if drive_fn is not None:
                spikes = drive_fn(step, spikes)
            # LÉSION : les neurones lésés ne tirent JAMAIS (ni récurrence,
            # ni drive) — leur colonne de keep_mask vaut 0.
            spikes = spikes * keep_mask
            spikes_sum += spikes
    return spikes_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="phases raccourcies (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.quick:
        global REFLEX_MS, SPONT_MS, DISCR_MS
        REFLEX_MS, SPONT_MS, DISCR_MS = 200, 200, 150

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print("=" * 78)
    print("  LÉSIONS VIRTUELLES — MaleCNS v1.0 (α-synapses Shiu, "
          f"Wsyn = {WSYN} mV)")
    print("=" * 78)

    W, neuron_ids, df, group_mask, grn_idx, mn9_idx, motor_mask, sens_pool = \
        load_everything()
    n = len(neuron_ids)
    print(f"\n[1] Connectome {n} neurones / {W.nnz} synapses | "
          f"{len(grn_idx)} GRN sucrées, {len(mn9_idx)} MN9, "
          f"{int(motor_mask.sum())} motoneurones, {len(sens_pool)} sensoriels "
          f"drivables")

    conds, sizes = build_conditions(rng, group_mask, n)
    B = len(conds)
    print(f"[2] {B} conditions simulées en parallèle :")
    for name, m in conds:
        print(f"    - {name:32s} {int(m.sum()):6d} neurones lésés")

    # keep_mask (n, B) float : 1 = vivant
    keep = np.stack([(~m).astype(np.float32) for _, m in conds], axis=1)
    keep_t = torch.tensor(keep)

    W_t = torch.sparse_csr_tensor(
        torch.tensor(W.indptr, dtype=torch.int64),
        torch.tensor(W.indices, dtype=torch.int64),
        torch.tensor(W.data, dtype=torch.float32) * WSYN,
        size=W.shape)

    grn_t = torch.tensor(grn_idx, dtype=torch.long)
    mn9_t = torch.tensor(mn9_idx, dtype=torch.long)
    motor_t = torch.tensor(np.where(motor_mask)[0], dtype=torch.long)
    stim_p = STIM_HZ * DT / 1000.0
    n_motor = int(motor_mask.sum())

    results = {"wsyn_mv": WSYN, "n_neurons": int(n), "n_synapses": int(W.nnz),
               "conditions": {}, "assays_ms": {"reflex": REFLEX_MS,
                                               "spont": SPONT_MS,
                                               "discrimination": DISCR_MS}}

    # --- Assay 1 : réflexe d'alimentation (Shiu) ---
    print(f"\n[3] Réflexe : Poisson {STIM_HZ:.0f} Hz sur GRN, {REFLEX_MS} ms")

    def reflex_drive(step, spikes):
        drive = (torch.rand(n, B) < stim_p).float()
        spikes[grn_t, :] = drive[grn_t, :]
        return spikes

    s = simulate_phase(W_t, keep_t, B, REFLEX_MS, drive_fn=reflex_drive,
                       seed=args.seed)
    mn9_rate = s[mn9_t].sum(dim=0).numpy() / len(mn9_idx) / (REFLEX_MS / 1000.0)
    base = mn9_rate[0]
    results["conditions"]["_reflex_intact_hz"] = float(base)

    # --- Assay 2 : spontané (Poisson d'ambiance 2 Hz sur les sensoriels +
    # bruit membranaire σ = 0,3 mV — le régime « au repos » du Tamagotchi) ---
    SPONT_HZ = 2.0
    spont_p = SPONT_HZ * DT / 1000.0
    sens_t = torch.tensor(sens_pool, dtype=torch.long)
    print(f"[4] Spontané : Poisson d'ambiance {SPONT_HZ:.0f} Hz sur "
          f"{len(sens_pool)} sensoriels + bruit σ = {NOISE_SIGMA} mV, "
          f"{SPONT_MS} ms")

    def spont_drive(step, spikes):
        drive = (torch.rand(n, B) < spont_p).float()
        spikes[sens_t, :] = drive[sens_t, :]
        return spikes

    s = simulate_phase(W_t, keep_t, B, SPONT_MS, drive_fn=spont_drive,
                       noise=True, seed=args.seed + 1)
    spont_all = s.sum(dim=0).numpy() / n / (SPONT_MS / 1000.0)
    spont_mot = s[motor_t].sum(dim=0).numpy() / n_motor / (SPONT_MS / 1000.0)

    # --- Assay 3 : discrimination sensorimotrice ---
    print(f"[5] Discrimination : {DISCR_SETS} sous-ensembles de 500 sensoriels "
          f"à {STIM_HZ:.0f} Hz, {DISCR_MS} ms chacun")
    n_sub = min(500, len(sens_pool))
    subsets = [rng.choice(sens_pool, size=n_sub, replace=False)
               for _ in range(DISCR_SETS)]
    patterns = np.zeros((B, DISCR_SETS, n_motor))
    gen = torch.Generator().manual_seed(args.seed + 2)
    for k, subset in enumerate(subsets):
        sub_t = torch.tensor(subset, dtype=torch.long)

        def discr_drive(step, spikes, sub_t=sub_t):
            drive = (torch.rand(n, B, generator=gen) < stim_p).float()
            spikes[sub_t, :] = drive[sub_t, :]
            return spikes

        s = simulate_phase(W_t, keep_t, B, DISCR_MS, drive_fn=discr_drive,
                           seed=args.seed + 10 + k)
        patterns[:, k, :] = s[motor_t].T.numpy() / (DISCR_MS / 1000.0)

    # --- Agrégation ---
    print("\n" + "=" * 78)
    print(f"  {'CONDITION':32s} {'lésés':>7} | {'MN9 Hz':>8} {'déficit':>8} | "
          f"{'spont Hz':>9} {'mot Hz':>7} | {'spécif.':>8}")
    print("-" * 78)
    for i, (name, _) in enumerate(conds):
        if i == 0:
            deficit = 0.0
        else:
            deficit = max(0.0, 1.0 - mn9_rate[i] / base) if base > 0 else np.nan
        # spécificité : 1 − corrélation de Pearson moyenne entre les 4 patterns
        P = patterns[i]
        if P.std() < 1e-9:
            spec = np.nan
        else:
            C = np.corrcoef(P)
            spec = float(1.0 - np.nanmean(C[np.triu_indices(DISCR_SETS, 1)]))
        results["conditions"][name] = {
            "lesioned": int(sizes[name]),
            "reflex_mn9_hz": float(mn9_rate[i]),
            "reflex_deficit": float(deficit) if np.isfinite(deficit) else None,
            "spont_hz_per_neuron": float(spont_all[i]),
            "spont_motor_hz": float(spont_mot[i]),
            "motor_specificity": spec if np.isfinite(spec) else None,
            "motor_patterns_hz": patterns[i].mean(axis=1).tolist(),
        }
        def fmt(x, pct=False):
            return "—" if x is None or not np.isfinite(x) else \
                f"{x:8.1%}" if pct else f"{x:8.1f}"
        print(f"  {name:32s} {sizes[name]:7d} | {fmt(mn9_rate[i])} "
              f"{fmt(deficit, pct=True) if i > 0 else '     —'} | "
              f"{fmt(spont_all[i])} {fmt(spont_mot[i])} | {fmt(spec)}")

    # Interprétation rapide
    print("-" * 78)
    if base > 0:
        order = sorted([(k, v) for k, v in results["conditions"].items()
                        if isinstance(v, dict)
                        and v["reflex_deficit"] is not None],
                       key=lambda kv: -kv[1]["reflex_deficit"])
        top = [f"{k} ({v['reflex_deficit']:.0%})" for k, v in order[:3]
               if v["reflex_deficit"] > 0.05]
        print(f"  Plus gros déficits sur le réflexe : "
              f"{'; '.join(top) if top else 'aucun > 5 %'}")

    out = DATA_DIR / "lesion_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[6] Résultats sauvegardés : {out} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
