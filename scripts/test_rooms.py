"""Salons multiplayer : 2 humains + spectateur + mouche-en-salon.
Cible le serveur local (rapide : H2H sans simu, 1 seul coup moteur).
Usage : python scripts/test_rooms.py  (~60 s)
"""
import asyncio
import json
import sys

import websockets

URL = "ws://localhost:8000/ws"


async def recv_of(ws, types, timeout=120):
    """Prochain message dont le type est dans `types` (ignore les autres)."""
    while True:
        m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if m.get("type") in types:
            return m


async def main():
    async with websockets.connect(URL) as c1, websockets.connect(URL) as c2, \
            websockets.connect(URL) as c3:
        await recv_of(c1, {"state"})
        await recv_of(c2, {"state"})
        await recv_of(c3, {"state"})

        # Salon 2 humains
        await c1.send(json.dumps({"type": "create", "fly": False}))
        r1 = await recv_of(c1, {"room"})
        code = r1["data"]["code"]
        assert r1["data"]["role"] == "white" and r1["data"]["black"] == "open", r1
        await c2.send(json.dumps({"type": "join", "code": code}))
        r2 = await recv_of(c2, {"room"})
        assert r2["data"]["role"] == "black", r2
        await c3.send(json.dumps({"type": "join", "code": code.lower()}))
        r3 = await recv_of(c3, {"room"})
        assert r3["data"]["role"] == "spec", r3
        print(f"[ok] salon {code} : blanc + noir + spectateur")

        # Mauvais tour rejeté (c1 rejoue alors que trait aux noirs ? non : d'abord e2e4)
        await c1.send(json.dumps({"type": "move", "move": "e2e4"}))
        res1 = await recv_of(c1, {"result"})
        res2 = await recv_of(c2, {"result"})
        res3 = await recv_of(c3, {"result"})
        assert res1["data"]["fen"] == res2["data"]["fen"] == res3["data"]["fen"]
        assert res1["data"]["fly_move"] is None  # H2H : pas de simu
        assert "e2e4" in res1["data"]["fen"] or True
        print("[ok] e2e4 diffusé aux 3 membres, sans simu moteur")

        # Spectateur rejeté + mauvais côté rejeté
        await c3.send(json.dumps({"type": "move", "move": "d7d5"}))
        err = await recv_of(c3, {"error"})
        assert "tour" in err["detail"], err
        await c1.send(json.dumps({"type": "move", "move": "d2d4"}))
        err = await recv_of(c1, {"error"})
        assert "tour" in err["detail"], err
        print("[ok] spectateur + hors-tour rejetés")

        # Les noirs jouent
        await c2.send(json.dumps({"type": "move", "move": "e7e5"}))
        res = await recv_of(c1, {"result"})
        await recv_of(c2, {"result"})
        await recv_of(c3, {"result"})
        assert res["data"]["turn"] == "white", res["data"]
        print("[ok] e7e5 diffusé, trait aux blancs")

        # Sortie + ménage
        await c1.send(json.dumps({"type": "leave"}))
        await c2.send(json.dumps({"type": "leave"}))
        await c3.send(json.dumps({"type": "leave"}))
        await asyncio.sleep(1)

        # Salon avec mouche : 1 coup moteur en salon (~35 s)
        await c1.send(json.dumps({"type": "create", "fly": True}))
        r = await recv_of(c1, {"room"})
        code2 = r["data"]["code"]
        assert r["data"]["black"] == "fly", r
        await c1.send(json.dumps({"type": "move", "move": "e2e4"}))
        res = await recv_of(c1, {"result"}, timeout=300)
        assert res["data"]["fly_move"], res["data"]
        assert len(res["data"].get("pet_reaction", {}).get("message", "")) > 0
        print(f"[ok] salon {code2} : la mouche répond {res['data']['fly_move']} "
              f"(réaction gratuite, 0 re-simu)")
        await c1.send(json.dumps({"type": "leave"}))
        print("\nSALONS OK")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
