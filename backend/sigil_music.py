"""
Musique générative sigillaire : chaque motif devient un instrument.
Mapping (déterministe, depuis les métriques mesurées de bench_sigil) :
  domHz  -> hauteur de base (le rythme propre du motif)
  burst  -> vélocité (texture : lisse vs hachée)
  MC     -> longueur de phrase (mémoire = structure)
  rate   -> densité (notes par temps)

Sortie : un .mid format 0 écrit à la main (zéro dépendance), avec un
marqueur texte par motif. Écouter = entendre la topologie.
"""
import struct

from bench_sigil import SigilCircuit, build_W, run_and_measure, PATTERNS


def _varlen(n: int) -> bytes:
    out = bytes([n & 0x7F])
    n >>= 7
    while n:
        out = bytes([(n & 0x7F) | 0x80]) + out
        n >>= 7
    return out


def phrase(pattern_id: int, seed: int = 0) -> list[tuple]:
    """8-16 événements (pitch, vel, dur_ticks) depuis les métriques live."""
    circ = SigilCircuit()
    nn = circ.native_n(pattern_id)
    n = nn if nn else 64
    W, _ = build_W(circ, pattern_id, n, seed=seed)
    rate, burst, dom, MC = run_and_measure(circ, W, T=1500, seed=seed)
    base = 48 + int(dom * 40) % 24
    vel = max(40, min(120, 60 + int(burst)))
    length = 8 + int(MC * 20)          # mémoire -> structure (8..12 notes)
    density = 1 if rate < 0.02 else 2  # activité -> densité
    evts, p = [], base
    step = (pattern_id % 5) - 2        # signature mélodique du motif
    for i in range(length):
        for _ in range(density):
            p = max(36, min(84, p + step + ((i * 7) % 3 - 1)))
            evts.append((p, vel, 240))
    return evts


def write_midi(path: str, pattern_ids=None):
    """Un .mid, une piste, un marqueur + une phrase par motif."""
    if pattern_ids is None:
        pattern_ids = list(range(len(PATTERNS)))
    track = b""
    track += b"\x00\xff\x51\x03\x07\xa1\x20"  # tempo 120
    for pid in pattern_ids:
        name = PATTERNS[pid]
        track += _varlen(0) + b"\xff\x06" + _varlen(len(name)) + name.encode()
        for pitch, vel, dur in phrase(pid):
            track += _varlen(0) + bytes([0x90, pitch, vel])
            track += _varlen(dur) + bytes([0x80, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + track)
    return path


if __name__ == "__main__":
    out = write_midi("_sigil_demo.mid", pattern_ids=[0, 2, 22, 24, 6])
    data = open(out, "rb").read()
    assert data[:4] == b"MThd" and b"flower" in data and b"iching" in data
    print(f"{out}: {len(data)} octets, 5 motifs (seal/ring/flower/iching/random)")
    import os
    os.remove(out)
    print("OK")