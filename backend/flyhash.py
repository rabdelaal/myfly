"""
FlyHash — recherche par similarité façon mouche (Dasgupta et al. 2017) :
le circuit olfactif de la drosophile fait du locality-sensitive hashing.

Ici : AnythingEncoder (texte -> pattern sensoriel) + k gagnants (winners)
-> hash binaire sparse. Similarité = recouvrement de Hamming.
Ultra-léger, déterministe, sans embeddings lourds.

Usage :
    from flyhash import FlyHash
    fh = FlyHash()
    fh.index(["doc1", "doc2", ...])
    print(fh.query("motif proche de doc1", top=2))  # [(idx, overlap), ...]
"""
import numpy as np
import torch

from encoding import AnythingEncoder


class FlyHash:
    # ponytail: 512 dims (pas 256) — en dessous, les collisions de trigrammes
    # sur textes courts dominent le classement.
    def __init__(self, n_sensory: int = 512, k: int = 26, seed: int = 7):
        self.enc = AnythingEncoder(n_sensory=n_sensory, seed=seed)
        self.k = k
        self.codes = None
        self.docs = []

    def _hash(self, text: str) -> np.ndarray:
        with torch.no_grad():
            c = self.enc.encode_options([text])[:, 0].numpy()
        top = np.argpartition(c, -self.k)[-self.k:]
        h = np.zeros(len(c), dtype=np.uint8)
        h[top] = 1
        return h

    def index(self, docs: list[str]):
        self.docs = list(docs)
        self.codes = np.stack([self._hash(d) for d in docs], axis=0)
        return self

    def query(self, text: str, top: int = 3):
        assert self.codes is not None, "index() d'abord"
        h = self._hash(text).astype(np.int32)
        # ponytail: produits/sommes en int32 — en uint8, -overlap déborde
        # (255 au lieu de -1) et argsort classe à l'envers.
        overlap = (self.codes.astype(np.int32) * h[None, :]).sum(axis=1)
        order = np.argsort(-overlap, kind="stable")[:top]
        return [(int(i), int(overlap[i])) for i in order]


if __name__ == "__main__":
    docs = [
        "the queen sacrifices for checkmate",
        "queen checkmate attack on the king",
        "the cat sleeps on the warm rug",
        "a dog naps on the carpet",
        "quantum entanglement of photons",
        "drosophila mushroom body memory",
    ]
    fh = FlyHash().index(docs)
    q = fh.query("queen checkmate sacrifice", top=2)
    print("query ->", [(docs[i], o) for i, o in q])
    assert {i for i, _ in q} == {0, 1}, q  # les 2 docs échecs en tête
    q2 = fh.query("cat rug sleep", top=1)
    print("query2 ->", docs[q2[0][0]])
    assert q2[0][0] in (2, 3), q2
    print("OK")