"""Test du mode « Cerveau en direct » : frames WS, hz par population, événement."""
import asyncio
import json

import urllib.request


def post(path, body=None):
    req = urllib.request.Request(
        "http://localhost:8000" + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def main():
    import websockets

    # Réveiller la mouche si besoin (sinon cerveau au repos complet)
    st = post("/api/pet/action", {"action": "wake"})
    print(f"[pet] réveil : {st['message']} (applied={st.get('applied')})")

    async with websockets.connect("ws://localhost:8000/ws/live") as ws:
        # Collecter 3 secondes de frames de base
        frames, total_spikes = 0, 0
        hz_last = None
        try:
            while frames < 90:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m["type"] == "live":
                    frames += 1
                    total_spikes += len(m["spikes"])
                    hz_last = m["hz"]
        except asyncio.TimeoutError:
            pass
        print(f"[live] {frames} frames en 3 s, {total_spikes} spikes cumulés, "
              f"dernier hz : {hz_last}")

        # Injecter un événement de soin pendant qu'on regarde
        post("/api/pet/action", {"action": "feed"})
        got_caption = None
        spike_burst = 0
        try:
            for _ in range(60):  # ~2 s
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m.get("caption"):
                    got_caption = m["caption"]
                if m.get("spikes"):
                    spike_burst += len(m["spikes"])
        except asyncio.TimeoutError:
            pass
        print(f"[live] après nourrissage : légende = {got_caption!r}, "
              f"{spike_burst} spikes en 2 s (bouffée de stimulus)")

    # Après déconnexion, la simulation doit s'arrêter (vérifié dans les logs)
    print("[live] déconnecté — la boucle doit s'arrêter toute seule")


asyncio.run(main())
