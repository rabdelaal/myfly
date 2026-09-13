"""
Convertit les fichiers Feather officiels MaleCNS v1.0 (Janelia) en un .npz
compact pour le backend.

Sources (https://male-cns.janelia.org/download/, CC-BY) :
  - body-annotations-male-cns-v1.0-minconf-0.5.feather   (superclass, somaLocation…)
  - body-neurotransmitters-male-cns-v1.0.feather         (NT par corps)
  - connectome-weights-male-cns-v1.0-minconf-0.5.feather (connectivité agrégée)

Lecture memory-mappée batch par batch + lookup vectorisé (searchsorted)
pour tenir dans une RAM modeste.

Usage : python prepare_data.py
"""
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
NPZ_PATH = DATA_DIR / "malecns.npz"

INHIBITORY = {"gaba", "glycine"}


def sc_is_sensory(sc):
    return sc is not None and "sensory" in sc


def sc_is_motor(sc):
    return sc is not None and ("motor" in sc or "efferent" in sc)


def sc_is_descending(sc):
    return sc is not None and "descending" in sc


def build_annotations():
    """bodyId → index ; positions (somaLocation) ; masques par superclass."""
    print("[prepare] Annotations…")
    ann = feather.read_table(DATA_DIR / "body-annotations.feather", memory_map=True)
    names = set(ann.schema.names)
    def _col(name):
        return ann[name].to_pylist() if name in names else [None] * ann.num_rows
    status = _col("status")
    superclass = _col("superclass")
    body_ids = _col("bodyId")
    soma = _col("somaLocation")
    # Colonnes papier Cell 2026 (A0) — optionnelles selon version du dump
    col_type = _col("type")
    col_dim = _col("dimorphism")
    col_fru = _col("fruDsx") if "fruDsx" in names else _col("fru_dsx")
    col_class = _col("class") if "class" in names else [None] * ann.num_rows
    col_subclass = _col("subclass")

    keep_ids, positions, sc_kept = [], [], []
    keep_type, keep_dim, keep_fru = [], [], []
    keep_class, keep_subclass = [], []
    for i, bid in enumerate(body_ids):
        if status[i] in ("Glia", "Unimportant", None):
            continue
        keep_ids.append(int(bid))
        loc = soma[i]
        positions.append([float(v) for v in loc] if loc else [0.0, 0.0, 0.0])
        sc_kept.append(superclass[i])
        keep_type.append(str(col_type[i]) if col_type[i] else "")
        keep_dim.append(str(col_dim[i]) if col_dim[i] else "")
        keep_fru.append(str(col_fru[i]) if col_fru[i] else "")
        keep_class.append(str(col_class[i]) if col_class[i] else "")
        keep_subclass.append(str(col_subclass[i]) if col_subclass[i] else "")

    ids = np.array(keep_ids, dtype=np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    id_to_row = {int(b): i for i, b in enumerate(keep_ids)}
    n = len(keep_ids)

    is_sensory = np.zeros(n, dtype=bool)
    is_motor = np.zeros(n, dtype=bool)
    is_descending = np.zeros(n, dtype=bool)
    for i, sc in enumerate(sc_kept):
        if sc_is_sensory(sc):
            is_sensory[i] = True
        elif sc_is_motor(sc):
            is_motor[i] = True
        elif sc_is_descending(sc):
            is_descending[i] = True

    print(f"[prepare] {n} neurones retenus | sensoriels {is_sensory.sum()}, "
          f"moteurs {is_motor.sum()}, descendants {is_descending.sum()}")
    extra = {
        "types": np.array(keep_type),
        "dimorphism": np.array(keep_dim),
        "frudsx": np.array(keep_fru),
        "cell_class": np.array(keep_class),
        "subclass": np.array(keep_subclass),
    }
    # Compte-rendu papier Cell Table 1 (prouvable : 1420 male-specific / 948 dimorphic)
    try:
        import collections
        print(f"[prepare] dimorphisme : {collections.Counter(keep_dim)}")
        print(f"[prepare] fruDsx : {collections.Counter(keep_fru).most_common(8)}")
    except Exception:
        pass
    return ids, sorted_ids, id_to_row, np.array(positions, dtype=np.float32), \
        is_sensory, is_motor, is_descending, extra


def build_nt_signs(n, sorted_ids):
    """Signe synaptique par index, d'après le NT prédit de plus haute confiance."""
    print("[prepare] Neurotransmetteurs…")
    t = feather.read_table(DATA_DIR / "body-neurotransmitters.feather",
                           memory_map=True)
    t = t.sort_by([("predicted_nt_confidence", "descending")])
    bodies = t["body"].to_numpy()
    nts = t["predicted_nt"].to_pylist()

    # Premier passage (confiance max) par corps, puis projection vectorisée
    best_body, best_nt = [], []
    seen = np.zeros(0, dtype=np.int64)
    seen_set = set()
    for b, x in zip(bodies, nts):
        b = int(b)
        if b in seen_set:
            continue
        seen_set.add(b)
        best_body.append(b)
        best_nt.append(-1.0 if x in INHIBITORY else 1.0)

    bb = np.array(best_body, dtype=np.int64)
    bs = np.array(best_nt, dtype=np.float32)
    signs = np.ones(n, dtype=np.float32)
    pos = np.searchsorted(sorted_ids, bb)
    pos = np.clip(pos, 0, n - 1)
    valid = sorted_ids[pos] == bb
    signs[pos[valid]] = bs[valid]
    print(f"[prepare] {len(seen_set)} corps annotés, "
          f"{int((signs < 0).sum())} inhibiteurs")
    return signs


def lookup(rows, sorted_ids):
    """bodyId → index (vectorisé, -1 si absent)."""
    pos = np.searchsorted(sorted_ids, rows)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    ok = sorted_ids[pos] == rows
    out = np.full(len(rows), -1, dtype=np.int32)
    out[ok] = pos[ok].astype(np.int32)
    return out


def build_weights(sorted_ids, signs):
    """Connectivité : lecture IPC memory-mappée, batches → COO accumulé."""
    path = DATA_DIR / "connectome-weights.feather"
    print("[prepare] Connectivité (streaming IPC)…")
    src = pa.memory_map(str(path), "r")
    reader = pa.ipc.open_file(src)
    names = reader.schema.names
    print(f"[prepare] colonnes : {names}")
    pre_col = "bodyId_pre" if "bodyId_pre" in names else names[0]
    post_col = "bodyId_post" if "bodyId_post" in names else names[1]
    w_col = "weight" if "weight" in names else names[2]

    pre_parts, post_parts, w_parts = [], [], []
    total = 0
    for bi in range(reader.num_record_batches):
        batch = reader.get_batch(bi)
        pre = lookup(batch.column(pre_col).to_numpy(), sorted_ids)
        post = lookup(batch.column(post_col).to_numpy(), sorted_ids)
        w = batch.column(w_col).to_numpy().astype(np.float32)
        ok = (pre >= 0) & (post >= 0)
        pre_parts.append(pre[ok])
        post_parts.append(post[ok])
        w_parts.append(w[ok] * signs[pre[ok]])
        total += batch.num_rows
        if (bi + 1) % 20 == 0:
            print(f"[prepare]   batch {bi + 1}/{reader.num_record_batches} "
                  f"({total} lignes)")
    src.close()

    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    w = np.concatenate(w_parts)
    print(f"[prepare] {total} lignes lues, {len(pre)} connexions retenues")
    W = sp.csr_matrix((w, (pre, post)),
                      shape=(len(sorted_ids), len(sorted_ids)))
    W.sum_duplicates()
    return W


def main():
    ids, sorted_ids, id_to_row, positions, is_sensory, is_motor, is_descending, extra = \
        build_annotations()
    signs = build_nt_signs(len(ids), sorted_ids)
    W = build_weights(sorted_ids, signs)

    print(f"[prepare] W : {W.shape[0]}×{W.shape[1]}, {W.nnz} synapses agrégées, "
          f"densité {W.nnz / (W.shape[0] ** 2) * 100:.2f} %")
    np.savez_compressed(
        NPZ_PATH,
        weights=W.data,
        indices=W.indices.astype(np.int32),
        indptr=W.indptr.astype(np.int64),
        neuron_ids=ids,
        positions=positions,
        is_sensory=is_sensory,
        is_motor=is_motor,
        is_descending=is_descending,
        types=extra["types"],
        dimorphism=extra["dimorphism"],
        frudsx=extra["frudsx"],
        cell_class=extra["cell_class"],
        subclass=extra["subclass"],
        format="csr",
    )
    print(f"[prepare] Sauvegardé dans {NPZ_PATH}")


if __name__ == "__main__":
    main()
