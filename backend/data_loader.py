"""
Chargement du connectome MaleCNS.
- Si les fichiers préparés existent (data/malecns.npz), on les charge.
- Sinon, on génère un connectome synthétique pour valider le pipeline.
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
NPZ_PATH = DATA_DIR / "malecns.npz"
ANN_PATH = DATA_DIR / "body-annotations.feather"


def _add_special_neurons(conn):
    """
    Identifie les neurones « spéciaux » du réflexe d'alimentation de Shiu et al.
    dans les annotations réelles : GRN du goût (tpGRN) et motoneurones MN9.
    Ajoute conn["sugar_grn_idx"] et conn["mn9_idx"] (indices dans W).
    """
    conn["sugar_grn_idx"] = np.zeros(0, dtype=np.int64)
    conn["mn9_idx"] = np.zeros(0, dtype=np.int64)
    if not ANN_PATH.exists():
        return
    try:
        import pyarrow.feather as feather
        ann = feather.read_table(ANN_PATH, memory_map=True)
        df = ann.select(["bodyId", "type", "status"]).to_pandas()
        ids = conn["neuron_ids"]
        sorted_ids = np.sort(ids)
        order = np.argsort(ids)

        def idx_of(body_ids):
            body_ids = np.asarray(body_ids, dtype=np.int64)
            pos = np.searchsorted(sorted_ids, body_ids)
            pos = np.clip(pos, 0, len(sorted_ids) - 1)
            ok = sorted_ids[pos] == body_ids
            out = np.full(len(body_ids), -1, dtype=np.int64)
            out[ok] = order[pos][ok]
            return out[out >= 0]

        types = df["type"].fillna("").astype(str)
        traced = df["status"] == "Traced"
        grn = df[types.str.contains("tpGRN", case=False) & traced]["bodyId"].to_numpy()
        mn9 = df[types.str.upper() == "MN9"]["bodyId"].to_numpy()
        conn["sugar_grn_idx"] = idx_of(grn)
        conn["mn9_idx"] = idx_of(mn9)
        print(f"[data] neurones spéciaux : {len(conn['sugar_grn_idx'])} GRN goût "
              f"(tpGRN), {len(conn['mn9_idx'])} MN9")
    except Exception as e:
        print(f"[data] annotations réelles indisponibles ({e}) — pas de GRN/MN9")


def _add_dimorphism_masks(conn, dimorphism=None, frudsx=None):
    """Masques papier Cell 2026 (A1). Entrées: arrays str alignés sur neuron_ids
    ou None (vieux npz / synthétique → zéros, jamais de crash).
    Expose is_male_specific / is_dimorphic / is_fru / is_dsx / is_hotspot."""
    n = len(conn["neuron_ids"])
    def _zeros():
        return np.zeros(n, dtype=bool)
    try:
        if dimorphism is None or frudsx is None:
            raise ValueError("annotations dimorphisme absentes")
        dim = np.asarray(dimorphism).astype(str)
        fru = np.asarray(frudsx).astype(str)
        conn["dimorphism"] = dim
        conn["frudsx"] = fru
        conn["is_male_specific"] = np.char.find(dim, "male-specific") >= 0
        # "potentially male-specific" contient aussi "male-specific" → les 4
        # catégories restent distinguables via conn["dimorphism"], le masque
        # large sert la visualisation, pas la vérité terrain (cf. rapport A5).
        conn["is_dimorphic"] = np.char.find(dim, "dimorphic") >= 0
        conn["is_fru"] = np.isin(fru, ["fru_high", "fru_low", "coexpress_high", "coexpress_low"])
        conn["is_dsx"] = np.isin(fru, ["dsx_high", "dsx_low", "coexpress_high", "coexpress_low"])
        conn["is_hotspot"] = conn["is_male_specific"] | conn["is_dimorphic"] | conn["is_fru"] | conn["is_dsx"]
        print(f"[data] dimorphisme : {int(conn['is_male_specific'].sum())} male-specific, "
              f"{int(conn['is_dimorphic'].sum())} dimorphic, "
              f"{int(conn['is_fru'].sum())} fru+, {int(conn['is_dsx'].sum())} dsx+")
    except Exception as e:
        conn["is_male_specific"] = _zeros()
        conn["is_dimorphic"] = _zeros()
        conn["is_fru"] = _zeros()
        conn["is_dsx"] = _zeros()
        conn["is_hotspot"] = _zeros()
        print(f"[data] masques dimorphisme vides ({e})")


def dn_maxflow_mask(conn, k: int = 256) -> np.ndarray:
    """Sous-ensemble DN à fort max-flow sensor→DN→moteur (RAPPORT A3.1).

    Heuristique prouvable sans le papier Data S1 : score(DN) =
    force_entrante(sensoriels) × force_sortante(vers moteurs), sur |W|.
    Retourne un masque booléen (n,) avec ≤k descendants. k<=0 → tous.
    Ablation prévue : DN-sélectionnés vs DN-bruts (cf. train_readout_looped)."""
    import numpy as np
    is_desc = np.asarray(conn["is_descending"], dtype=bool)
    desc_idx = np.nonzero(is_desc)[0]
    if len(desc_idx) == 0 or k is not None and k >= len(desc_idx) or (k or 0) <= 0:
        return is_desc.copy()
    W = conn["W"].tocsr()
    is_sens = np.asarray(conn["is_sensory"], dtype=bool)
    is_mot = np.asarray(conn["is_motor"], dtype=bool)
    inflow = np.asarray(np.abs(W[:, desc_idx])[is_sens].sum(axis=0)).ravel()
    outflow = np.asarray(np.abs(W[desc_idx][:, is_mot]).sum(axis=1)).ravel()
    score = inflow * outflow
    # Toujours garder les DN actifs même à score nul (stabilité)
    take = np.argsort(-score, kind="stable")[:max(int(k), 1)]
    mask = np.zeros(W.shape[0], dtype=bool)
    mask[desc_idx[take]] = True
    print(f"[data] DN max-flow : {int(mask.sum())}/{len(desc_idx)} retenus "
          f"(score max {float(score.max()):.1f})")
    return mask


def load_connectome(force_synthetic: bool = False):
    """
    Retourne un dict :
        - W : csr_matrix (n_neurons, n_neurons) signée (excitateur +, inhibiteur -)
        - neuron_ids : array (n_neurons,) des bodyId
        - types : array (n_neurons,) des types cellulaires
        - positions : array (n_neurons, 3) des positions 3D
        - is_sensory, is_motor, is_descending : masques booléens
    """
    if NPZ_PATH.exists() and not force_synthetic:
        print(f"[data] Chargement du connectome réel depuis {NPZ_PATH}")
        data = np.load(NPZ_PATH, allow_pickle=True)
        if "format" in data and str(data["format"]) == "csr":
            # Format MaleCNS réel : CSR pré-calculé (economique en RAM)
            W = sp.csr_matrix(
                (data["weights"], data["indices"], data["indptr"]),
                shape=(len(data["neuron_ids"]), len(data["neuron_ids"])),
            )
        else:
            # Format historique (COO)
            W = sp.csr_matrix(
                (data["weights"], (data["pre_idx"], data["post_idx"])),
                shape=(len(data["neuron_ids"]), len(data["neuron_ids"])),
            )
        # Normalisation pour le mode "current" (moyenne |w| = 1). Le mode
        # "alpha" (Shiu) reconstruit les poids bruts via mean_abs_weight :
        # w_alpha = w_norm × mean_w × 20 × Wsyn (w_norm = brut / (mean×20)).
        mean_w = float(np.abs(W.data).mean())
        if mean_w > 0:
            W.data = W.data / (mean_w * 20.0)
        conn = {
            "W": W,
            "neuron_ids": data["neuron_ids"],
            "types": data["types"] if "types" in data else
                     np.array([f"type_{i}" for i in range(len(data["neuron_ids"]))]),
            "positions": data["positions"],
            "is_sensory": data["is_sensory"],
            "is_motor": data["is_motor"],
            "is_descending": data["is_descending"],
            "mean_abs_weight": mean_w,
        }
        _add_special_neurons(conn)
        _add_dimorphism_masks(conn, data["dimorphism"] if "dimorphism" in data else None,
                              data["frudsx"] if "frudsx" in data else None)
        return conn

    print("[data] Génération d'un connectome SYNTHÉTIQUE (démo)")
    conn = _synthetic_connectome()
    _add_special_neurons(conn)  # sans vrai npz/annotations → vide
    _add_dimorphism_masks(conn)
    return conn


def _synthetic_connectome(n_neurons: int = 5000, sparsity: float = 0.005, seed: int = 42):
    """
    Connectome synthétique avec structure : 30% sensoriel, 50% interneurones, 20% moteur.
    Les connexions sont aléatoires mais biaisées vers les couches profondes.
    """
    rng = np.random.default_rng(seed)

    n_sensory = int(0.3 * n_neurons)
    n_motor = int(0.2 * n_neurons)
    n_inter = n_neurons - n_sensory - n_motor

    n_edges = int(n_neurons * n_neurons * sparsity)
    pre = rng.integers(0, n_neurons, n_edges)
    post = rng.integers(0, n_neurons, n_edges)

    # Biais : les sensoriels projettent surtout vers les interneurones
    # les interneurones vers les moteurs
    mask_ss = (pre < n_sensory) & (post < n_sensory)
    pre = pre[~mask_ss]
    post = post[~mask_ss]
    n_edges = len(pre)

    # Signe : 80% excitateur, 20% inhibiteur
    signs = rng.choice([1, -1], size=n_edges, p=[0.8, 0.2])
    weights = rng.gamma(shape=2.0, scale=1.0, size=n_edges) * signs

    W = sp.csr_matrix((weights, (pre, post)), shape=(n_neurons, n_neurons))

    is_sensory = np.zeros(n_neurons, dtype=bool)
    is_sensory[:n_sensory] = True
    is_motor = np.zeros(n_neurons, dtype=bool)
    is_motor[n_sensory + n_inter:] = True
    is_descending = np.zeros(n_neurons, dtype=bool)
    is_descending[n_sensory + n_inter - 100: n_sensory + n_inter] = True

    positions = rng.normal(size=(n_neurons, 3)) * 100
    types = np.array([f"type_{i % 50}" for i in range(n_neurons)])
    neuron_ids = np.arange(n_neurons)

    return {
        "W": W,
        "neuron_ids": neuron_ids,
        "types": types,
        "positions": positions,
        "is_sensory": is_sensory,
        "is_motor": is_motor,
        "is_descending": is_descending,
        "mean_abs_weight": float(np.abs(W.data).mean()) if W.nnz else 1.0,
    }
