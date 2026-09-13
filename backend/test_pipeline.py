"""Test de bout en bout : perf des coups batchés, octopamine, WebSocket."""
import asyncio
import json
import time

import urllib.request

BASE = "http://localhost:8000"


def post(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


async def ws_test():
    import websockets
    frames = 0
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Attendre l'état initial envoyé par le serveur, puis jouer un coup légal
        first = json.loads(await ws.recv())
        assert first["type"] == "state", first
        legal = first["data"]["legal_moves"]
        move = legal[0]
        t0 = time.time()
        await ws.send(json.dumps({"type": "move", "move": move}))
        result = None
        while result is None:
            msg = json.loads(await ws.recv())
            if msg["type"] == "frame":
                frames += 1
            elif msg["type"] == "result":
                result = msg["data"]
            elif msg["type"] == "error":
                raise SystemExit(f"WS error: {msg['detail']}")
        dt = time.time() - t0
        print(f"[WS] coup {move} -> mouche {result['fly_move']} en {dt:.2f}s, "
              f"{frames} frames streamées en direct")
        print(f"[WS] octopamine: {result['pet']['octopamine']}, "
              f"bruit échecs: {result['pet']['chess_noise']}")


async def main():
    # 1. Perf REST : 5 coups consécutifs (batchés)
    post("/api/new", {})
    times = []
    fly = None
    for move in ["e2e4", "d2d4", "g1f3", "c2c4", "e2e3"]:
        st = get("/api/state")
        if move not in st["legal_moves"]:
            continue
        t0 = time.time()
        r = post("/api/move", {"move": move})
        times.append(time.time() - t0)
        fly = r.get("fly_move")
    print(f"[REST] {len(times)} coups joués, temps moyen {sum(times)/len(times):.2f}s "
          f"(batch: 20 candidats simulés en parallèle), dernier coup mouche: {fly}")

    # 2. Octopamine : humeur haute -> gain > 1 ; on la rend grognonne
    st = get("/api/pet")
    print(f"[pet] humeur={st['mood']} octopamine={st['octopamine']} "
          f"bien-être={st['wellbeing']}")

    # 3. WebSocket avec streaming
    await ws_test()


asyncio.run(main())
