"""
⚗️ Labo des lésions virtuelles : définit les populations lésionnables
(annotations MaleCNS v1.0 — celles du papier Cell 2026), applique/efface la
lésion sur les cerveaux de la mouche (Tamagotchi + live) et mesure le déficit
du réflexe d'alimentation (assay validé de exp_feeding_reflex / exp_lesions).

Les déficits « prédits » affichés dans l'UI viennent de l'expérience batchée
exp_lesions.py (data/lesion_results.json) ; le test à la demande mesure le
réflexe sur le cerveau réel de la mouche, dans les mêmes conditions.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "lesion_results.json"

REFLEX_MS = 600          # fenêtre de mesure (≈44 s sur MaleCNS à 73 ms/pas)
STIM_HZ = 100.0          # drive Poisson des GRN sucrées (Shiu et al.)

# (clé API, libellé UI, source) — source : "cell_class" | "dimorphism" |
# "frudsx" (colonnes npz), "descending" (masque), "feather:superclass:VALUE"
GROUPS = [
    ("mb", "Corps du champignon (MB)", ("cell_class", "Kenyon_Cell")),
    ("cx", "Complexe central (CX)", ("cell_class", "CX")),
    ("dimorphic", "Dimorphes mâle/femelle", ("dimorphism", None)),
    ("frudsx", "fruitless / doublesex", ("frudsx", None)),
    ("descending", "Neurones descendants", ("descending", None)),
    ("optic", "Lobes optiques", ("feather:superclass", "ol_intrinsic")),
    ("random4k", "Témoin aléatoire (~4k)", ("__random__", 4000)),
]

# noms de conditions utilisés par exp_lesions.py → clés API (déficits prédits)
EXP_NAME_TO_KEY = {
    "corps du champignon (MB)": "mb",
    "complexe central (CX)": "cx",
    "dimorphes mâle/femelle": "dimorphic",
    "fruitless/doublesex": "frudsx",
    "neurones descendants": "descending",
    "lobes optiques": "optic",
    "témoin aléatoire ~4k": "random4k",
}


class LesionLab:
    def __init__(self, conn):
        self.conn = conn
        self.current: str | None = None
        self.baseline_reflex_hz: float | None = None
        self.REFLEX_MS = REFLEX_MS
        self._baseline_lock = threading.Lock()
        self._rng = np.random.default_rng(42)
        self._group_idx = self._build_groups(conn)
        self._predicted = self._load_predicted_deficits()

    # --- Construction des groupes (une seule fois) ---
    def _build_groups(self, conn) -> dict:
        t0 = time.time()
        n = conn["W"].shape[0]
        npz = np.load(DATA_DIR / "malecns.npz", allow_pickle=True)
        cols = {}
        for col in ("cell_class", "dimorphism", "frudsx"):
            cols[col] = (np.asarray(npz[col]).astype(str)
                         if col in npz.files else np.array([""] * n))
        # superclass n'est pas dans le npz : lecture feather une fois
        super_arr = np.array([""] * n, dtype=object)
        try:
            import pyarrow.feather as feather
            ann = feather.read_table(DATA_DIR / "body-annotations.feather",
                                     memory_map=True)
            df = ann.select(["bodyId", "superclass"]).to_pandas()
            ids = conn["neuron_ids"]
            sorted_ids = np.sort(ids)
            order = np.argsort(ids)
            pos = np.searchsorted(sorted_ids, df["bodyId"].to_numpy())
            ok = (pos < len(sorted_ids)) & \
                 (sorted_ids[np.clip(pos, 0, len(sorted_ids) - 1)] ==
                  df["bodyId"].to_numpy())
            rows = order[pos[ok]]
            super_arr[rows] = df["superclass"].fillna("").to_numpy()[ok]
        except Exception as e:
            print(f"[lab] superclass indisponible ({e}) — groupe 'optic' vide")

        groups = {}
        for key, _name, (src, val) in GROUPS:
            m = np.zeros(n, dtype=bool)
            if src == "__random__":
                m[self._rng.choice(n, size=int(val), replace=False)] = True
            elif src == "descending":
                m[np.asarray(conn["is_descending"], dtype=bool)] = True
            elif src.startswith("feather:"):
                m[super_arr == val] = True
            else:
                # colonne npz : toute valeur annotée (val None) ou valeur exacte
                m = (cols[src] != "") if val is None else (cols[src] == val)
            groups[key] = np.nonzero(m)[0]
        print(f"[lab] {len(groups)} groupes de lésion construits "
              f"({time.time() - t0:.1f} s) : "
              + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
        return groups

    def _load_predicted_deficits(self) -> dict:
        """Déficits réflexe mesurés par l'expérience batchée, par clé API."""
        out = {}
        try:
            res = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            base = res["conditions"]["intact (témoin)"]["reflex_mn9_hz"]
            for name, c in res["conditions"].items():
                key = EXP_NAME_TO_KEY.get(name)
                if key and c.get("reflex_deficit") is not None:
                    out[key] = {"deficit": c["reflex_deficit"],
                                "mn9_hz": c["reflex_mn9_hz"],
                                "window_ms": res.get("assays_ms", {})
                                .get("reflex", 800)}
            out["_intact_hz"] = base
        except Exception as e:
            print(f"[lab] résultats exp_lesions indisponibles ({e})")
        return out

    # --- API ---
    def info(self) -> dict:
        groups = []
        for key, name, _src in GROUPS:
            pred = self._predicted.get(key)
            groups.append({
                "key": key, "name": name, "n": int(len(self._group_idx[key])),
                "predicted_deficit": pred["deficit"] if pred else None,
            })
        return {
            "current": self.current,
            "groups": groups,
            "baseline_reflex_hz": self.baseline_reflex_hz,
            "reflex_ms": REFLEX_MS,
            "predicted_intact_hz": self._predicted.get("_intact_hz"),
        }

    def indices(self, key: str) -> np.ndarray:
        return self._group_idx[key]

    def apply(self, key: str, brains: list) -> int:
        idx = self._group_idx[key]
        for brain in brains:
            brain.set_lesion(idx)
        self.current = key
        return int(len(idx))

    def clear(self, brains: list) -> None:
        for brain in brains:
            brain.clear_lesion()
        self.current = None

    # --- Mesure du réflexe d'alimentation (assay Shiu) ---
    @torch.no_grad()
    def measure_reflex(self, brain, ms: int = REFLEX_MS) -> float:
        """Taux MN9 (Hz) pendant un drive Poisson 100 Hz des GRN sucrées.
        Utilise le cerveau de la mouche (état réinitialisé, restauré après)."""
        mn9 = self.conn.get("mn9_idx")
        if mn9 is None or len(mn9) == 0:
            raise RuntimeError("MN9 introuvables (annotations absentes)")
        mn9_t = torch.tensor(np.asarray(mn9, dtype=np.int64),
                             device=brain.device)
        brain.reset(1)
        brain.set_sugar_drive(STIM_HZ, ms)
        total = 0.0
        for _ in range(ms):
            brain.step(None)
            total += float(brain.spikes[mn9_t].sum().item())
        brain.set_sugar_drive(0.0, 0)
        brain._sugar_drive_left = 0
        return total / len(mn9) / (ms / 1000.0)

    def ensure_baseline(self, brain) -> float:
        """Mesure la référence intacte une seule fois (thread-safe). La lésion
        éventuellement active est suspendue puis restaurée : la référence doit
        refléter un cerveau intact même si l'utilisateur lèse pendant la mesure
        (thread de démarrage vs clic UI)."""
        if self.baseline_reflex_hz is not None:
            return self.baseline_reflex_hz
        with self._baseline_lock:
            if self.baseline_reflex_hz is not None:
                return self.baseline_reflex_hz
            saved = self.current
            if saved is not None:
                brain.clear_lesion()
            try:
                print("[lab] mesure de la référence réflexe (cerveau intact)…")
                hz = self.measure_reflex(brain)
            finally:
                if saved is not None:
                    brain.set_lesion(self._group_idx[saved])
            self.baseline_reflex_hz = hz
            print(f"[lab] référence intacte : {hz:.1f} Hz")
        return self.baseline_reflex_hz
