"""
Réplication de l'expérience d'alimentation de Shiu et al. (Nature 2024) sur
NOTRE connectome MaleCNS v1.0.

Protocole du papier :
  - Stimuler les GRN sucrées en Poisson à 100 Hz pendant 1 s
  - Mesurer la décharge du motoneurone MN9 (procobiscis = alimentation)
  - Calibrer l'unique paramètre libre Wsyn pour que 100 Hz donne ~80 %
    de la réponse motrice maximale (leur valeur : 0,275 mV/synapse)

Modèle LIF exact de Shiu et al. :
  - V_rest = V_reset = -52 mV, V_th = -45 mV
  - α-synapses : g decroît avec τ_syn = 5 ms, incrément +w à chaque spike
    présynaptique, effet après un délai de 1,8 ms
  - réfractaire absolu 2,2 ms
  - poids = nombre de synapses agrégées × signe(NT) × Wsyn
  - zéro décharge de base (aucun bruit)

Usage : python exp_feeding_reflex.py
"""
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path

torch.set_num_threads(4)

DATA_DIR = Path(__file__).parent.parent / "data"
DT = 1.0                 # ms
V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
TAU_M = 20.0             # ms (hypothèse standard ; τ_syn et délai = Shiu)
TAU_SYN = 5.0            # ms
DELAY_STEPS = 2          # 1,8 ms → 2 pas de 1 ms
REFRACTORY_STEPS = 3     # 2,2 ms → 3 pas
STIM_HZ = 100.0
SIM_MS = 1000
TRIALS = 5
WSYN_SWEEP = [0.05, 0.1, 0.175, 0.275, 0.45, 0.8, 1.5]  # mV ; 0.275 = Shiu


def load_signed_connectome():
    """npz : poids = nb de synapses agrégées × signe(NT), NON normalisés."""
    d = np.load(DATA_DIR / "malecns.npz")
    W = sp.csr_matrix((d["weights"], d["indices"], d["indptr"]),
                      shape=(len(d["neuron_ids"]), len(d["neuron_ids"])))
    return W, d["neuron_ids"]


def main():
    print("=" * 78)
    print("  RÉPLICATION SHIU ET AL. 2024 — RÉFLEXE D'ALIMENTATION sur MaleCNS")
    print("=" * 78)

    # --- 1. Identification des neurones ---
    import pyarrow.feather as feather
    ann = feather.read_table(DATA_DIR / "body-annotations.feather", memory_map=True)
    df = ann.select(["bodyId", "type", "instance", "superclass", "status"]).to_pandas()

    mn9 = df[df["type"].fillna("").str.contains("MN9", case=False)]
    grn = df[df["type"].fillna("").str.contains("tpGRN", case=False)]
    print(f"\n[1] MN9 : {len(mn9)} neurones {[x for x in mn9['instance']]}, "
          f"GRN goût (tpGRN) : {len(grn)} neurones")

    W, neuron_ids = load_signed_connectome()
    n = W.shape[0]
    sorted_ids = np.sort(neuron_ids)
    order = np.argsort(neuron_ids)

    def idx_of(body_ids):
        pos = np.searchsorted(sorted_ids, body_ids)
        ok = (pos < len(sorted_ids)) & (sorted_ids[np.clip(pos, 0, len(sorted_ids) - 1)] == body_ids)
        # repasser en index d'origine (neuron_ids non trié)
        orig = order[pos]
        out = np.full(len(body_ids), -1, dtype=np.int64)
        out[ok] = orig[ok]
        return out

    mn9_idx = idx_of(mn9["bodyId"].to_numpy())
    mn9_idx = mn9_idx[mn9_idx >= 0]
    grn_idx = idx_of(grn["bodyId"].to_numpy())
    grn_idx = grn_idx[grn_idx >= 0]
    # ne garder que les GRN ayant des synapses sortantes
    has_out = np.asarray(W[grn_idx].sum(axis=1)).ravel() != 0
    grn_idx = grn_idx[has_out]
    print(f"    dans le connectome : {len(mn9_idx)} MN9, {len(grn_idx)} GRN actives")

    Wt = torch.sparse_csr_tensor(
        torch.tensor(W.indptr, dtype=torch.int64),
        torch.tensor(W.indices, dtype=torch.int64),
        torch.tensor(W.data, dtype=torch.float32),
        size=W.shape)

    mn9_t = torch.tensor(mn9_idx, dtype=torch.int64)
    grn_t = torch.tensor(grn_idx, dtype=torch.int64)
    n_grn = len(grn_idx)

    decay_syn = float(np.exp(-DT / TAU_SYN))
    stim_p = STIM_HZ * DT / 1000.0    # proba de spike Poisson par pas

    # --- 2. Balayage de Wsyn ---
    print(f"\n[2] Balayage Wsyn — {STIM_HZ:.0f} Hz Poisson sur {n_grn} GRN sucrées, "
          f"{SIM_MS} ms, {TRIALS} essais batchés")
    print(f"    {'Wsyn (mV)':>10} | {'taux MN9 (Hz)':>13} | {'GRN (Hz)':>9}")
    rates = []
    rng = np.random.default_rng(42)

    for Wsyn in WSYN_SWEEP:
        Wk = Wt * Wsyn
        B = TRIALS
        V = torch.full((n, B), V_REST)
        g = torch.zeros(n, B)              # conductances α (mV)
        refr = torch.zeros(n, B, dtype=torch.int32)
        delay_buf = [torch.zeros(n, B) for _ in range(DELAY_STEPS)]
        mn9_total, grn_total = 0.0, 0.0
        with torch.no_grad():
            for step in range(SIM_MS // int(DT)):
                # délai 1,8 ms : les spikes agissent avec 2 pas de retard
                delayed = delay_buf.pop(0)
                delay_buf.append(spikes.clone() if step > 0 else torch.zeros(n, B))
                inc = torch.sparse.mm(Wk, delayed)
                g = g * decay_syn + inc
                I_syn = g
                V = V + (-(V - V_REST) + I_syn) * (DT / TAU_M)
                sp = (V >= V_TH) & (refr <= 0)
                V = torch.where(sp, torch.full_like(V, V_RESET), V)
                refr = torch.where(sp, torch.full_like(refr, REFRACTORY_STEPS, dtype=torch.int32),
                                   (refr - 1).clamp(min=0))
                spikes = sp.float()
                # Stimulation à 100 Hz EXACTE : les GRN sucrées sont pilotées
                # (leurs spikes sont REMPLACÉS par le train de Poisson, comme
                # dans Shiu et al. — sinon la récurrence les fait tirer à 250 Hz)
                drive = (torch.rand(n, B) < stim_p).float()
                spikes[grn_t, :] = drive[grn_t, :]
                mn9_total += float(spikes[mn9_t, :].sum())
                grn_total += float(spikes[grn_t, :].sum())
        rate_mn9 = mn9_total / len(mn9_idx) / (SIM_MS / 1000.0) / B
        rate_grn = grn_total / n_grn / (SIM_MS / 1000.0) / B
        rates.append(rate_mn9)
        print(f"    {Wsyn:10.3f} | {rate_mn9:13.1f} | {rate_grn:9.1f}")

    # --- 3. Calibration 80 % ---
    rates = np.array(rates)
    rate_max = rates.max()
    sweep = np.array(WSYN_SWEEP)
    target = 0.8 * rate_max
    # interpolation linéaire sur la montée
    cal = None
    for i in range(len(sweep) - 1):
        if rates[i] < target <= rates[i + 1]:
            cal = float(np.interp(target, [rates[i], rates[i + 1]], [sweep[i], sweep[i + 1]]))
            break
    print(f"\n[3] Réponse maximale : {rate_max:.1f} Hz (Wsyn = {sweep[int(rates.argmax())]} mV)")
    if cal:
        print(f"    Wsyn calibré pour 80 % du max : {cal:.3f} mV/synapse "
              f"(Shiu et al. : 0.275 mV)")
        print(f"    écart avec Shiu : ×{cal / 0.275:.2f}")
    else:
        print(f"    80 % du max non atteint dans la plage balayée — élargir WSYN_SWEEP")
    np.save(DATA_DIR / "feeding_dose_response.npy",
            np.stack([sweep, rates]))
    print(f"\n    courbe dose-réponse sauvegardée : feeding_dose_response.npy")


if __name__ == "__main__":
    main()
