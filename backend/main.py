"""
API FastAPI : expose les endpoints pour jouer contre la mouche.
WebSocket /ws : les spikes du connectome sont streamés en temps réel pendant
que la mouche réfléchit à son coup.
"""
import asyncio
import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import numpy as np
from pathlib import Path

from data_loader import load_connectome
from brain import FlyBrain
from encoding import BoardEncoder
from readout import load_readout, LinearReadout
from chess_engine import FlyChessEngine
from tamagotchi import FlyPet
from live import LiveSim
from lab import LesionLab
# flyintel est un package à la racine du repo (hors backend/) : on l'ajoute au
# path pour pouvoir importer websearch (outil d'apprentissage web du modèle).
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from flyintel import websearch

# --- État global (un seul cerveau, chargé au démarrage) ---
STATE = {
    "brain": None,
    "encoder": None,
    "readout": None,
    "engine": None,
    "connectome": None,
    "pet": None,
    "live": None,
    "lab": None,
    "board": chess.Board(),
    "history": [],
    "rooms": {},  # code -> {board, history, members, roles, fly_side, queues}
}

# Une seule mouche (cerveau avec état) : les coups sont sérialisés même
# entre salons (corrige aussi la course solo à 2 onglets).
_ENGINE_LOCK = asyncio.Lock()

_ROOM_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ"  # sans I/L/O confus


def _new_room_code() -> str:
    import random
    while True:
        code = "".join(random.choice(_ROOM_ALPHABET) for _ in range(4))
        if code not in STATE["rooms"]:
            return code


def _room_public(room, code: str) -> dict:
    roles = list(room["roles"].values())
    return {
        "code": code,
        "fen": room["board"].fen(),
        "turn": "white" if room["board"].turn else "black",
        "white": "fly" if room["fly_side"] == "white" else ("human" if "white" in roles else "open"),
        "black": "fly" if room["fly_side"] == "black" else ("human" if "black" in roles else "open"),
        "spectators": sum(1 for r in roles if r == "spec"),
        "game_over": room["board"].is_game_over(),
    }

# Verrou du test réflexe : évite de corrompre l'état du cerveau si un coup
# d'échecs est simulé en même temps (les deux tournent en threadpool).
_lab_test_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Chargement du connectome...")
    conn = load_connectome()

    # Régime prod (alpha/Shiu par défaut) : construction partagée train/inférence
    from brain import build_prod_brain
    synapse_model = os.environ.get("FLY_SYNAPSE_MODEL", "alpha")
    wsyn = float(os.environ.get("FLY_WSYN", "0.164"))  # mV/synapse, calibré chez nous
    brain = build_prod_brain(conn, synapse_model=synapse_model, wsyn=wsyn)
    # Encodage : classic (défaut, Shiu intact) ou modal (A2, FLY_ENCODER=modal)
    encoder_kind = os.environ.get("FLY_ENCODER", "classic").lower()
    if encoder_kind == "modal":
        from encoding import ModalBoardEncoder
        encoder = ModalBoardEncoder(n_sensory=int(conn["is_sensory"].sum()))
    else:
        encoder = BoardEncoder(n_sensory=int(conn["is_sensory"].sum()))

    # Sélection DN (A3) : all (défaut) ou maxflow (FLY_DN=maxflow, FLY_DN_K=256)
    dn_mode = os.environ.get("FLY_DN", "all").lower()
    dn_k = int(os.environ.get("FLY_DN_K", "256"))
    dn_mask = None
    if dn_mode == "maxflow":
        from data_loader import dn_maxflow_mask
        full = dn_maxflow_mask(conn, k=dn_k)
        desc_global = np.nonzero(np.asarray(conn["is_descending"]))[0]
        dn_idx = np.isin(desc_global, np.nonzero(full)[0])
        dn_mask = dn_idx  # booléen sur les descendants (moteur intact)
    n_motor = int(conn["is_motor"].sum())
    n_descending = int(dn_mask.sum()) if dn_mask is not None else int(conn["is_descending"].sum())

    # Readout : linear (défaut) ou looped (B2, FLY_READOUT=looped)
    readout_kind = os.environ.get("FLY_READOUT", "linear").lower()
    looped_path = Path(__file__).parent / "readout_looped.pt"
    readout_path = Path(__file__).parent / "readout.pt"
    readout = None
    if readout_kind == "looped" and looped_path.exists():
        from readout_looped import load_looped
        readout, looped_meta = load_looped(str(looped_path), brain.device)
        if dn_mask is None and looped_meta.get("dn_mask") is not None:
            dn_mask = np.asarray(looped_meta["dn_mask"], dtype=bool)
            n_descending = int(dn_mask.sum())
        print(f"[startup] LoopedReadout chargé (d_in={readout.cell.merge.in_features - readout.cell.d_h}).")
    elif readout_path.exists():
        readout = load_readout(str(readout_path), n_motor, n_descending, brain.device)
        print("[startup] Readout entraîné chargé.")
    else:
        readout = LinearReadout(n_motor, n_descending).to(brain.device)
        print("[startup] ⚠️ Readout non entraîné (poids aléatoires).")

    engine = FlyChessEngine(brain, encoder, readout, dn_mask=dn_mask)
    pet = FlyPet(brain, n_sensory=int(conn["is_sensory"].sum()))
    brain_kwargs = dict(synapse_model=brain.synapse_model)
    if brain.synapse_model == "alpha":
        brain_kwargs.update(weight_scale=brain.weight_scale, V_rest=brain.V_rest,
                            V_th=brain.V_th, V_reset=brain.V_reset,
                            tau_syn=brain.tau_syn, delay_steps=brain.delay_steps,
                            refractory_steps=brain.refractory_steps)
    live = LiveSim(conn, pet, brain_kwargs=brain_kwargs)
    lab = LesionLab(conn)

    # Référence réflexe (cerveau intact) mesurée en tâche de fond : prête
    # ~1 min après le démarrage, sans bloquer l'API.
    threading.Thread(target=lambda: lab.ensure_baseline(brain),
                     daemon=True, name="lab-baseline").start()

    STATE.update(
        brain=brain, encoder=encoder, readout=readout, engine=engine, connectome=conn,
        pet=pet, live=live, lab=lab,
    )
    print(f"[startup] Prêt. {conn['W'].shape[0]} neurones, "
          f"{conn['W'].nnz} synapses, device={brain.device}, "
          f"synapses={synapse_model}"
          + (f" (Wsyn={wsyn} mV)" if synapse_model == "alpha" else "")
          + f", encodeur={encoder_kind}, readout={readout_kind}, dn={dn_mode}"
          + (f"(k={dn_k})" if dn_mode == "maxflow" else "")
          + f", hotspots={int(np.asarray(conn.get('is_hotspot', [])).sum())}"
          f", humeur de la mouche : {pet.mood()}")
    yield


app = FastAPI(title="FlyChess API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schémas ---
class MoveRequest(BaseModel):
    move: str  # UCI (ex: "e2e4")


class NewGameRequest(BaseModel):
    player_color: str = "white"


class PetActionRequest(BaseModel):
    action: str  # feed | pet | clean | sleep | wake | courtship | threat | study


class LabLesionRequest(BaseModel):
    key: str


class LearnRequest(BaseModel):
    query: str
    num: int = 3
    dir: str = "knowledge"


class ReservoirRequest(BaseModel):
    pattern: str = "flower"
    series: list[float] = []
    horizon: int = 1


class IEGenerateRequest(BaseModel):
    world: str = "ledger"  # ledger | roster
    n_events: int = 40
    invalid_rate: float = 0.12
    seed: int = 0
    op: str = ""  # défaut selon le monde


# --- Endpoints ---
@app.get("/api/info")
def info():
    import os as _os
    brain = STATE["brain"]
    conn = STATE["connectome"]
    resp = {
        "n_neurons": int(brain.n),
        "n_synapses": int(brain.n_synapses),
        "device": brain.device,
        "readout_trained": (Path(__file__).parent / "readout.pt").exists(),
        "readout_looped": (Path(__file__).parent / "readout_looped.pt").exists(),
        "synapse_model": brain.synapse_model,
        "encoder": _os.environ.get("FLY_ENCODER", "classic"),
        "readout_kind": _os.environ.get("FLY_READOUT", "linear"),
        "dn_mode": _os.environ.get("FLY_DN", "all"),
        "n_hotspot": int(np.asarray(conn.get("is_hotspot", [])).sum()),
        "n_male_specific": int(np.asarray(conn.get("is_male_specific", [])).sum()),
        "n_fru": int(np.asarray(conn.get("is_fru", [])).sum()),
        "n_dsx": int(np.asarray(conn.get("is_dsx", [])).sum()),
    }
    if brain.synapse_model == "alpha":
        resp["wsyn_mv"] = float(brain.weight_scale / (STATE["connectome"]["mean_abs_weight"] * 20.0))
        sg = getattr(brain, "_sugar_idx", None)
        resp["n_sugar_grn"] = int(sg.numel()) if sg is not None else 0
    return resp


@app.get("/api/pet")
def pet_status():
    return STATE["pet"].status()


@app.post("/api/pet/action")
def pet_action(req: PetActionRequest):
    try:
        reaction = STATE["pet"].do_action(req.action)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # L'événement est injecté comme stimulus visible dans le cerveau en direct
    if (STATE["live"] is not None and reaction.get("applied")
            and reaction.get("message")):
        STATE["live"].event(req.action, reaction["message"])
    return {**reaction, "pet": STATE["pet"].status()}


# --- 📚 Apprentissage web (le modèle apprend de nouvelles données à chaud) ---

_learn_lock = threading.Lock()


@app.post("/api/learn")
def learn_endpoint(req: LearnRequest):
    """Recherche web (Exa si EXA_API_KEY, sinon DuckDuckGo), récupère et
    persiste les résultats en markdown — données ré-ingérables par le modèle."""
    if not req.query.strip():
        raise HTTPException(400, "query vide")
    if not 1 <= req.num <= 10:
        raise HTTPException(400, "num entre 1 et 10")
    with _learn_lock:  # évite d'écraser knowledge/ par des appels concurrents
        try:
            saved = websearch.learn(
                req.query, dir=req.dir, num=req.num, sleep_s=1.0)
        except Exception as e:
            raise HTTPException(502, f"apprentissage échoué: {e}")
    return {"query": req.query, "learned": len(saved), "files": saved}


@app.get("/api/knowledge")
def knowledge_list():
    """Ce que le modèle a appris : [{file, title, source, size, mtime,
    excerpt}], plus récent d'abord. Boucle la boucle de /api/learn."""
    with _learn_lock:
        items = websearch.list_learned("knowledge")
    return {"count": len(items), "items": items}


@app.get("/api/sigils")
def sigils_list():
    """Le codex : métriques mesurées des 56 motifs (symbolcodex/metrics.json).
    Alimente l'onglet vitrine Sigils."""
    import json
    p = Path(__file__).resolve().parent.parent / "symbolcodex" / "metrics.json"
    if not p.exists():
        raise HTTPException(503, "metrics.json absent (lancer bench_sigil.py)")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(502, f"lecture codex échouée: {e}")


@app.get("/api/search")
def search_memory(q: str = "", top: int = 3):
    """Mémoire sémantique : retrouve les lectures pertinentes (FlyHash).
    Ferme la boucle apprendre → indexer → rappeler → citer."""
    if not q.strip():
        raise HTTPException(400, "q vide")
    top = max(1, min(int(top), 10))
    with _learn_lock:
        from flyhash import KnowledgeIndex
        try:
            return KnowledgeIndex("knowledge").search(q, top=top)
        except Exception as e:
            raise HTTPException(502, f"recherche échouée: {e}")


@app.post("/api/reservoir")
def reservoir_endpoint(req: ReservoirRequest):
    """Réservoir sigillaire en service : fit sur la 1re moitié de `series`,
    prédit à `horizon`, retourne R². Garde-fous CPU (série ≤ 2000 pts)."""
    from reservoir_kit import SigilReservoir, r2
    from bench_sigil import PATTERNS
    if req.pattern not in PATTERNS:
        raise HTTPException(400, f"pattern inconnu (choix: {len(PATTERNS)} motifs)")
    if not 20 <= len(req.series) <= 2000:
        raise HTTPException(400, "series : 20..2000 points")
    if not 1 <= req.horizon <= 50:
        raise HTTPException(400, "horizon : 1..50")
    try:
        u = np.asarray(req.series, dtype=np.float32)
        y = np.concatenate([np.zeros(req.horizon), u[:-req.horizon]])
        cut = len(u) // 2
        # Washout adaptatif : le fixe (20) vidait les petites séries
        # (ex. 20 pts -> 10 prédits - 20 washout = 0 point -> crash).
        washout = max(5, min(20, cut // 4, (len(u) - cut) // 4))
        if cut <= washout or len(u) - cut <= washout:
            raise HTTPException(400, "series trop courte pour ce washout")
        r = SigilReservoir(pattern=req.pattern, n=32, washout=washout)
        r.fit(u[:cut], y[:cut])
        p = r.predict(u[cut:])
        score = r2(y[cut:], p)
    except Exception as e:
        raise HTTPException(502, f"réservoir échoué: {e}")
    return {"pattern": req.pattern, "horizon": req.horizon,
            "r2": round(score, 4), "n": len(req.series)}


@app.post("/api/ie/generate")
def ie_generate(req: IEGenerateRequest):
    """Générateur IE à la demande (style EvolveScaler miniature) : un monde
    exécutable produit histoire + question + réponse + checklist."""
    from ie_worlds import LedgerWorld, RosterWorld
    if req.world not in ("ledger", "roster"):
        raise HTTPException(400, "world: ledger | roster")
    if not 10 <= req.n_events <= 300:
        raise HTTPException(400, "n_events : 10..300")
    if not 0.0 <= req.invalid_rate <= 0.5:
        raise HTTPException(400, "invalid_rate : 0..0.5")
    try:
        w = LedgerWorld(req.seed) if req.world == "ledger" else RosterWorld(req.seed)
        w.gen(req.n_events, invalid_rate=req.invalid_rate)
        ops = ("total", "top", "owes") if req.world == "ledger" else ("duty", "hours")
        op = req.op if req.op in ops else ops[req.seed % len(ops)]
        q, a, chk = w.ask(op)
    except Exception as e:
        raise HTTPException(502, f"génération IE échouée: {e}")
    return {"world": req.world, "op": op, "context": w.render(),
            "question": q, "answer": a, "checklist": chk}


# --- ⚗️ Labo des lésions virtuelles ---

_lab_display_cache: dict = {}


def _display_positions(group_idx: np.ndarray) -> list:
    """Indices (dans le sous-ensemble affiché de /api/connectome) des neurones
    d'un groupe lésé — pour les griser côté WebGL."""
    if _connectome_cache is None:
        return []
    cache_key = (id(group_idx), len(_connectome_cache["indices"]))
    hit = _lab_display_cache.get(cache_key)
    if hit is None:
        displayed = np.asarray(_connectome_cache["indices"])
        order = {int(g): p for p, g in enumerate(displayed.tolist())}
        hit = sorted({order[int(g)] for g in group_idx if int(g) in order})
        _lab_display_cache[cache_key] = hit
    return hit


@app.get("/api/lab")
def lab_status():
    info = STATE["lab"].info()
    if STATE["lab"].current:
        info["display_indices"] = _display_positions(
            STATE["lab"].indices(STATE["lab"].current))
    return info


@app.post("/api/lab/lesion")
def lab_lesion(req: LabLesionRequest):
    lab = STATE["lab"]
    try:
        n = lab.apply(req.key, [STATE["brain"], STATE["live"].brain])
    except KeyError:
        raise HTTPException(400, f"Groupe de lésion inconnu : {req.key}")
    name = next((g["name"] for g in lab.info()["groups"]
                 if g["key"] == req.key), req.key)
    if STATE["live"] is not None:
        STATE["live"].event("lab",
                            f"⚗️ Lésion : {name} ({n} neurones silenciés)")
    resp = lab_status()
    resp.update({"lesioned": n, "name": name})
    return resp


@app.post("/api/lab/clear")
def lab_clear():
    lab = STATE["lab"]
    had = lab.current
    lab.clear([STATE["brain"], STATE["live"].brain])
    if had and STATE["live"] is not None:
        STATE["live"].event("lab", "🩹 La mouche guérit : lésion levée")
    resp = lab_status()
    resp["cleared"] = bool(had)
    return resp


@app.post("/api/lab/test-reflex")
def lab_test_reflex():
    """
    Mesure le réflexe d'alimentation (drive 100 Hz des GRN → MN9) sur le
    cerveau de la mouche, lésé ou non. ≈45 s sur MaleCNS (600 pas à ~73 ms).
    """
    lab = STATE["lab"]
    if not _lab_test_lock.acquire(blocking=False):
        raise HTTPException(409, "Un test réflexe est déjà en cours")
    try:
        baseline = lab.ensure_baseline(STATE["brain"])
        mn9_hz = lab.measure_reflex(STATE["brain"])
        deficit = max(0.0, 1.0 - mn9_hz / baseline) if baseline > 0 else None
    finally:
        _lab_test_lock.release()
    if STATE["live"] is not None:
        pct = f" (−{deficit:.0%})" if deficit else ""
        STATE["live"].event(
            "lab", f"🧪 Réflexe : {mn9_hz:.0f} Hz{pct} — "
                   f"lésion : {lab.current or 'aucune'}")
    return {
        "mn9_hz": round(mn9_hz, 1),
        "baseline_hz": round(baseline, 1),
        "deficit": round(deficit, 3) if deficit is not None else None,
        "current": lab.current,
        "reflex_ms": lab.REFLEX_MS,
    }


_connectome_cache: dict | None = None  # payload /api/connectome calculé une fois


@app.get("/api/connectome")
def connectome(max_neurons: int = 5000):
    """
    Positions 3D + catégorie + synapses, pour la visualisation WebGL.
    Coûteux (26M synapses) → calculé une seule fois puis servi du cache.
    """
    global _connectome_cache
    if _connectome_cache is None:
        _connectome_cache = _build_connectome_payload(max_neurons)
    return _connectome_cache


def _build_connectome_payload(max_neurons: int) -> dict:
    conn = STATE["connectome"]
    n = conn["W"].shape[0]
    if n > max_neurons:
        idx = np.linspace(0, n - 1, max_neurons).astype(int)
    else:
        idx = np.arange(n)

    positions = np.asarray(conn["positions"])[idx]
    # Normaliser dans un cube [-5, 5] pour l'affichage
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    scale = 10.0 / max((hi - lo).max(), 1e-9)
    positions = (positions - (lo + hi) / 2) * scale

    category = np.zeros(len(idx), dtype=int)  # 0 = interneurone
    category[np.asarray(conn["is_sensory"])[idx]] = 1
    category[np.asarray(conn["is_motor"])[idx]] = 2
    category[np.asarray(conn["is_descending"])[idx]] = 3
    # Hotspots dimorphes fru/dsx (A4) : surlignés dans le théâtre
    hotspot = np.zeros(len(idx), dtype=int)
    if "is_hotspot" in conn:
        hotspot[np.asarray(conn["is_hotspot"])[idx]] = 1

    # Sous-ensemble de synapses (les plus fortes) pour le rendu « Cerveau en
    # direct » : lignes qui s'allument quand leurs deux neurones tirent.
    edges = _sample_edges(conn, idx, n_edges=1500)

    return {
        "n_total": int(n),
        "indices": idx.tolist(),
        "positions": positions.round(3).tolist(),
        "category": category.tolist(),
        "hotspot": hotspot.tolist(),
        "edges": edges,
    }


def _sample_edges(conn, kept_idx: np.ndarray, n_edges: int = 1500) -> dict:
    """Échantillonne n_edges synapses parmi les plus fortes DONT les deux
    neurones font partie du sous-ensemble affiché."""
    W = conn["W"].tocoo()
    nnz = len(W.data)
    if nnz == 0:
        return {"a": [], "b": [], "sign": []}
    kept_mask = np.zeros(conn["W"].shape[0], dtype=bool)
    kept_mask[kept_idx] = True
    pos_of = {int(g): p for p, g in enumerate(kept_idx.tolist())}

    # Candidats : les deux extrémités affichées (masque vectorisé sur 26M)
    both = kept_mask[W.row] & kept_mask[W.col]
    cand = np.nonzero(both)[0]
    if len(cand) > n_edges * 3:
        rng = np.random.default_rng(7)
        cand = rng.choice(cand, size=n_edges * 3, replace=False)
    top = cand[np.argsort(-np.abs(W.data[cand]))][:n_edges]

    a, b, sign = [], [], []
    for k in top:
        pre, post, w = int(W.row[k]), int(W.col[k]), float(W.data[k])
        if pre != post:
            a.append(pos_of[pre])
            b.append(pos_of[post])
            sign.append(1 if w > 0 else 0)
    return {"a": a, "b": b, "sign": sign}


@app.post("/api/new")
def new_game(req: NewGameRequest):
    STATE["board"] = chess.Board()
    STATE["history"] = []
    return {"fen": STATE["board"].fen(), "turn": "white"}


@app.get("/api/state")
def state():
    board = STATE["board"]
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "legal_moves": [m.uci() for m in board.legal_moves],
        "is_game_over": board.is_game_over(),
        "game_over": board.is_game_over(),  # alias : le frontend lit game_over
        "result": board.result() if board.is_game_over() else None,
    }


_FRAME_SENTINEL = object()


def _do_player_move(move_str: str, frame_callback=None, board=None,
                     history=None, use_engine: bool = True) -> dict:
    """
    Joue le coup du joueur + la réponse de la mouche, et renvoie le payload.
    frame_callback : si fourni, les frames du connectome sont streamées au fil
    de la simulation ; sinon elles sont collectées et renvoyées dans le payload.
    board/history : échiquier du salon (défaut : partie solo globale).
    use_engine=False : pas de réponse de la mouche (humain vs humain,
    réaction texte du pet, 0 simu).
    """
    board = board if board is not None else STATE["board"]
    history = history if history is not None else STATE["history"]
    pet = STATE["pet"]

    # Le Tamagotchi d'abord : décroissance + refus si elle dort
    pet.apply_decay()
    if pet.state["sleeping"]:
        raise HTTPException(409, "La mouche dort. Réveille-la depuis l'onglet Mouche 💤")

    try:
        move = chess.Move.from_uci(move_str)
    except Exception:
        raise HTTPException(400, f"Coup invalide : {move_str}")
    if move not in board.legal_moves:
        raise HTTPException(400, f"Coup illégal : {move_str}")

    # 1. Le joueur joue
    board.push(move)
    history.append(move.uci())

    if board.is_game_over():
        return {
            "fly_move": None,
            "fen": board.fen(),
            "game_over": True,
            "result": board.result(),
            "frames": [],
            "pet": pet.status(),
        }

    if not use_engine:
        # Humain vs humain : la mouche spectate (réaction texte, 0 simu)
        reaction = pet.on_chess_move(fast=True)
        if STATE["live"] is not None:
            STATE["live"].event("chess", f"♟️ {move.uci()} — {reaction['message']}")
        return {
            "fly_move": None,
            "fen": board.fen(),
            "turn": "white" if board.turn else "black",
            "game_over": False,
            "result": None,
            "frames": [],
            "pet": pet.status(),
            "pet_reaction": reaction,
        }

    # 2. La mouche répond (son humeur brouille son évaluation des coups)
    STATE["engine"].noise = pet.chess_noise()
    collected = [] if frame_callback is None else None
    callback = frame_callback if frame_callback is not None else collected.append
    fly_move, scores = STATE["engine"].choose_move(board, frame_callback=callback)
    if fly_move is None:
        return {
            "fly_move": None,
            "fen": board.fen(),
            "game_over": True,
            "result": "draw",
            "frames": [],
            "pet": pet.status(),
        }

    board.push(fly_move)
    history.append(fly_move.uci())

    # 3. Le mini-jeu a des effets sur la mouche (réaction SANS simu :
    # l'activité motrice du coup gagnant vient du batch moteur)
    reaction = pet.on_chess_move(motor=STATE["engine"]._last_winner_motor)
    if STATE["live"] is not None:
        STATE["live"].event("chess",
                            f"♟️ Coup joué : {fly_move.uci()} — {reaction['message']}")

    return {
        "fly_move": fly_move.uci(),
        "fen": board.fen(),
        "scores": scores,
        "frames": collected or [],
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "pet": pet.status(),
        "pet_reaction": reaction,
    }


@app.post("/api/move")
def play_move(req: MoveRequest):
    return _do_player_move(req.move)


@app.get("/api/rooms")
def list_rooms():
    """Salons ouverts : code, occupation, trait."""
    return {"rooms": [_room_public(r, c) for c, r in STATE["rooms"].items()]}


def _room_of(ws):
    for code, room in STATE["rooms"].items():
        if ws in room["members"]:
            return code, room
    return None, None


async def _room_broadcast(room, msg: dict):
    for q in list(room["queues"]):
        _safe_put(q, msg)


def _safe_put(q: asyncio.Queue, msg: dict):
    """put_nowait qui ne lève jamais (clients lents perdent des messages)."""
    try:
        q.put_nowait(msg)
    except Exception:
        pass


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Solo (inchangé) : {"type": "move", "move": "e2e4"} → frames + résultat.
    Salons : {"type": "create", "fly": true} → {"type": "room", ...} ;
    {"type": "join", "code": "AB12"} (noir si libre sinon spectateur) ;
    {"type": "leave"} ; {"type": "new"} reset. Les coups en salon sont
    diffusés à tous les membres (frames + résultat).
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue(maxsize=64)  # diffusion salon → moi
    my_room: str | None = None

    async def send_room(code: str):
        room = STATE["rooms"].get(code)
        if room is not None:
            roles = room["roles"]
            await websocket.send_json({
                "type": "room",
                "data": {**_room_public(room, code),
                         "role": roles.get(websocket, "spec")},
            })

    def leave_room():
        nonlocal my_room
        if my_room is None:
            return
        room = STATE["rooms"].get(my_room)
        if room is not None:
            room["members"].discard(websocket)
            room["roles"].pop(websocket, None)
            room["queues"].discard(inbox)
            if not room["members"]:
                STATE["rooms"].pop(my_room, None)
        my_room = None

    async def solo_move(move_str: str):
        queue: asyncio.Queue = asyncio.Queue()

        def push_frame(frame):
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def work():
            try:
                payload = _do_player_move(move_str, frame_callback=push_frame)
                payload["frames"] = []  # déjà streamées
                return payload
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _FRAME_SENTINEL)

        async with _ENGINE_LOCK:
            result_task = loop.run_in_executor(None, work)
            while True:
                item = await queue.get()
                if item is _FRAME_SENTINEL:
                    break
                await websocket.send_json({"type": "frame", "frame": item})
            try:
                payload = await result_task
                await websocket.send_json({"type": "result", "data": payload})
            except HTTPException as e:
                await websocket.send_json({"type": "error", "detail": e.detail})

    async def room_move(code: str, room: dict, move_str: str):
        role = room["roles"].get(websocket)
        side = "white" if room["board"].turn else "black"
        if role not in ("white", "black") or role != side:
            await websocket.send_json({"type": "error",
                                       "detail": "Pas ton tour (ou spectateur)"})
            return
        # La mouche répond si le trait passe à son camp après notre coup
        fly_responds = room["fly_side"] is not None and room["fly_side"] != role
        queues = list(room["queues"])

        def fanout(frame):
            item = {"kind": "frame", "frame": frame}
            for q in queues:
                loop.call_soon_threadsafe(_safe_put, q, item)

        def work():
            return _do_player_move(
                move_str, frame_callback=fanout,
                board=room["board"], history=room["history"],
                use_engine=fly_responds)

        # use_engine : la mouche répond si le trait est à son camp
        async with _ENGINE_LOCK:
            result_task = loop.run_in_executor(None, work)
            try:
                payload = await result_task
            except HTTPException as e:
                await _room_broadcast(room, {"kind": "error", "detail": e.detail})
                return
        payload["frames"] = []  # déjà diffusées en direct
        await _room_broadcast(room, {"kind": "result", "data": payload})

    try:
        await websocket.send_json({"type": "state", "data": state()})
        recv_task = None
        inbox_task = None
        while True:
            recv_task = asyncio.create_task(websocket.receive_json())
            inbox_task = asyncio.create_task(inbox.get())
            done, pending = await asyncio.wait(
                {recv_task, inbox_task}, return_when=asyncio.FIRST_COMPLETED)
            if inbox_task in done:
                item = inbox_task.result()
                recv_task.cancel()
                try:
                    await recv_task
                except (asyncio.CancelledError, Exception):
                    pass
                kind = item.get("kind")
                if kind == "frame":
                    await websocket.send_json({"type": "frame", "frame": item["frame"]})
                elif kind == "result":
                    await websocket.send_json({"type": "result", "data": item["data"]})
                elif kind == "room":
                    await websocket.send_json({"type": "room", "data": item["data"]})
                elif kind == "error":
                    await websocket.send_json({"type": "error", "detail": item["detail"]})
                continue
            # Message du client
            inbox_task.cancel()
            try:
                await inbox_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                msg = recv_task.result()
            except WebSocketDisconnect:
                break
            mtype = msg.get("type")
            if mtype == "move":
                code, room = _room_of(websocket)
                if room is None:
                    await solo_move(msg["move"])
                else:
                    await room_move(code, room, msg["move"])
            elif mtype == "create":
                leave_room()
                code = _new_room_code()
                fly = bool(msg.get("fly", True))
                room = {"board": chess.Board(), "history": [],
                        "members": {websocket}, "roles": {websocket: "white"},
                        "fly_side": "black" if fly else None, "queues": {inbox}}
                STATE["rooms"][code] = room
                my_room = code
                await send_room(code)
            elif mtype == "join":
                leave_room()
                code = str(msg.get("code", "")).upper().strip()
                room = STATE["rooms"].get(code)
                if room is None:
                    await websocket.send_json({"type": "error", "detail": "Salon inconnu"})
                    continue
                roles = room["roles"]
                taken = set(roles.values())
                if room["fly_side"] != "black" and "black" not in taken:
                    roles[websocket] = "black"
                elif "white" not in taken:
                    roles[websocket] = "white"
                else:
                    roles[websocket] = "spec"
                room["members"].add(websocket)
                room["queues"].add(inbox)
                my_room = code
                await _room_broadcast(room, {"kind": "room",
                                             "data": _room_public(room, code)})
                await send_room(code)
            elif mtype == "leave":
                code, room = _room_of(websocket)
                leave_room()
                if room is not None:
                    await _room_broadcast(room, {"kind": "room",
                                                 "data": _room_public(room, code)})
                await websocket.send_json({"type": "result", "data": state()})
            elif mtype == "new":
                code, room = _room_of(websocket)
                if room is None:
                    STATE["board"] = chess.Board()
                    STATE["history"] = []
                    await websocket.send_json({"type": "result", "data": state()})
                else:
                    room["board"] = chess.Board()
                    room["history"] = []
                    await _room_broadcast(room, {"kind": "room",
                                                 "data": _room_public(room, code)})
            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        code, room = _room_of(websocket)
        leave_room()
        if room is not None and code in STATE["rooms"]:
            try:
                loop.create_task(_room_broadcast(room, {"kind": "room",
                                                        "data": _room_public(room, code)}))
            except Exception:
                pass


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """
    Cerveau en direct : flux continu de frames de spikes (30 fps) tant que le
    client reste connecté. La simulation ne tourne que s'il y a un spectateur.
    """
    await websocket.accept()
    live = STATE["live"]
    if live is None:
        await websocket.close()
        return
    queue = live.subscribe()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        live.unsubscribe(queue)


@app.post("/api/undo")
def undo():
    board = STATE["board"]
    if len(STATE["history"]) >= 2:
        board.pop()
        board.pop()
        STATE["history"] = STATE["history"][:-2]
    return {"fen": board.fen()}


@app.get("/api/hint")
def hint():
    board = STATE["board"]
    move, scores = STATE["engine"].choose_move(board)
    return {"hint": move.uci() if move else None, "scores": scores}
