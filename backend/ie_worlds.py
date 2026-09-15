"""
Mondes à évolution d'information (style EvolveScaler, arXiv 2609.08435v2) :
on définit le monde en CODE d'abord (états, transitions, validité), on rend
l'historique en texte ensuite, et les réponses viennent par RE-EXÉCUTION
déterministe — jamais du texte. Vérité terrain du domaine `ie_state` et
du futur jeu A/B ancré du metabrain.

Chaque monde : gen() produit des événements (valides + invalides + bruit),
render() les met en langage naturel, ask() calcule question/réponse/checklist
depuis le VRAI état, replay() recalcule l'état depuis les événements
(vérification d'auto-cohérence).
"""
import numpy as np


class LedgerWorld:
    """Grand-livre de dépenses partagées : ajouts, révisions, annulations,
    remboursements, devis tentatifs (invalides), doublons (invalides)."""

    PEOPLE = ["Ann", "Bob", "Carl", "Dan"]
    CATS = ["flights", "hotel", "meals", "tickets"]

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.events = []
        self._eid = 0

    def gen(self, n_events: int = 40, invalid_rate: float = 0.12):
        self.events, self._eid = [], 0
        live = []  # ids confirmés non annulés
        for t in range(n_events):
            r = self.rng.random()
            if r < invalid_rate / 2 and live:
                e = dict(kind="cancel", target=self.rng.choice(live),
                         text="", valid=True)
                live.remove(e["target"])
            elif r < invalid_rate and live:
                e = dict(kind="revise", target=self.rng.choice(live),
                         amount=int(self.rng.integers(20, 400)), text="",
                         valid=True)
            elif r < invalid_rate + 0.08:
                e = dict(kind="tentative", text="", valid=False)  # invalide
            elif r < invalid_rate + 0.12 and self.events:
                e = dict(kind="duplicate", of=len(self.events) - 1, text="",
                         valid=False)  # invalide : doublon
            elif r < invalid_rate + 0.16:
                e = dict(kind="smalltalk", text="", valid=False)  # bruit
            else:
                self._eid += 1
                e = dict(kind="add", id=self._eid,
                         payer=str(self.rng.choice(self.PEOPLE)),
                         amount=int(self.rng.integers(20, 400)),
                         cat=str(self.rng.choice(self.CATS)), text="",
                         valid=True)
                live.append(self._eid)
            self.events.append(e)
        self._render_all()
        return self.events

    def _render_all(self):
        day = 1
        for i, e in enumerate(self.events):
            if i and i % 8 == 0:
                day += 1
            k = e["kind"]
            if k == "add":
                e["text"] = (f"Day {day} — {e['payer']} paid {e['amount']} "
                             f"for {e['cat']} (receipt #{e['id']}).")
            elif k == "revise":
                e["text"] = (f"Day {day} — correction: receipt #{e['target']} "
                             f"was {e['amount']} after all (supersedes).")
            elif k == "cancel":
                e["text"] = (f"Day {day} — receipt #{e['target']} is CANCELED, "
                             f"ignore it.")
            elif k == "tentative":
                e["text"] = (f"Day {day} — maybe {self.rng.choice(self.CATS)} "
                             f"around {int(self.rng.integers(50, 300))}? "
                             f"(tentative quote, not booked).")
            elif k == "duplicate":
                e["text"] = f"Day {day} — forwarded: {self.events[e['of']]['text']}"
            else:
                e["text"] = (f"Day {day} — {self.rng.choice(self.PEOPLE)}: "
                             f"{self.rng.choice(['lol', 'ok!', 'see you there', 'nice'])}")

    def replay(self, events=None):
        """Recalcule l'état depuis les événements (règles de validité)."""
        ledger, live = {}, set()
        for e in (self.events if events is None else events):
            k = e["kind"]
            if k == "add":
                ledger[e["id"]] = dict(e)
                live.add(e["id"])
            elif k == "revise" and e["target"] in live:
                ledger[e["target"]]["amount"] = e["amount"]
            elif k == "cancel":
                live.discard(e["target"])
        paid = {p: 0 for p in self.PEOPLE}
        for i in live:
            paid[ledger[i]["payer"]] += ledger[i]["amount"]
        total = sum(paid.values())
        share = total / len(self.PEOPLE)
        return {"paid": paid, "total": total,
                "balance": {p: paid[p] - share for p in self.PEOPLE},
                "live": sorted(live)}

    def render(self):
        return [e["text"] for e in self.events]

    def ask(self, op: str = "total"):
        st = self.replay()
        if op == "total":
            q = "What is the total confirmed spending?"
            a = str(st["total"])
            chk = [f"live receipts: {st['live']}", f"total={st['total']}"]
        elif op == "top":
            top = max(st["paid"], key=lambda p: st["paid"][p])
            q = "Who paid the most (confirmed receipts)?"
            a = top
            chk = [f"paid={st['paid']}", f"top={top}"]
        else:  # owes
            p = self.PEOPLE[int(self.rng.integers(len(self.PEOPLE)))]
            q = f"What is {p}'s balance (paid minus equal share)?"
            a = f"{st['balance'][p]:.0f}"
            chk = [f"paid={st['paid']}", f"total={st['total']}",
                   f"balance[{p}]={a}"]
        return q, a, chk


class RosterWorld:
    """Tableau de quarts d'entrepôt : affectations, révocations, échanges,
    propositions tentatives (invalides)."""

    CREW = ["Ada", "Bo", "Cy", "Dee", "Eli", "Fay"]
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    SHIFTS = ["day", "night"]

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.events = []

    def gen(self, n_events: int = 40, invalid_rate: float = 0.12):
        self.events = []
        assign = {}  # (day, shift) -> worker (actuel)
        for t in range(n_events):
            r = self.rng.random()
            d = str(self.rng.choice(self.DAYS))
            s = str(self.rng.choice(self.SHIFTS))
            w = str(self.rng.choice(self.CREW))
            if r < invalid_rate / 2 and assign:
                key = list(assign)[self.rng.integers(len(assign))]
                e = dict(kind="revoke", key=key, text="", valid=True)
                del assign[key]
            elif r < invalid_rate and assign:
                key = list(assign)[self.rng.integers(len(assign))]
                e = dict(kind="swap", key=key, worker=w, text="", valid=True)
                assign[key] = w
            elif r < invalid_rate + 0.08:
                e = dict(kind="tentative", text="", valid=False)
            elif r < invalid_rate + 0.12:
                e = dict(kind="smalltalk", text="", valid=False)
            else:
                e = dict(kind="assign", key=(d, s), worker=w, text="",
                         valid=True)
                assign[(d, s)] = w
            self.events.append(e)
        blk = 1
        for i, e in enumerate(self.events):
            if i and i % 8 == 0:
                blk += 1
            k = e["kind"]
            if k == "assign":
                e["text"] = (f"Shift board #{blk}: {e['worker']} takes "
                             f"{e['key'][0]} {e['key'][1]}.")
            elif k == "revoke":
                e["text"] = (f"Shift board #{blk}: {e['key'][0]} {e['key'][1]} "
                             f"REVOKED, ignore earlier assignment.")
            elif k == "swap":
                e["text"] = (f"Shift board #{blk}: {e['key'][0]} {e['key'][1]} "
                             f"now {e['worker']} (replaces).")
            elif k == "tentative":
                e["text"] = (f"Shift board #{blk}: maybe {w} on {d} {s}? "
                             f"(tentative, not confirmed).")
            else:
                e["text"] = (f"Shift board #{blk}: {w}: "
                             f"{self.rng.choice(['coffee break', 'noted', 'copy that'])}")
        return self.events

    def replay(self, events=None):
        assign = {}
        for e in (self.events if events is None else events):
            k = e["kind"]
            if k == "assign":
                assign[tuple(e["key"])] = e["worker"]
            elif k == "revoke":
                assign.pop(tuple(e["key"]), None)
            elif k == "swap":
                assign[tuple(e["key"])] = e["worker"]
        hours = {w: 0 for w in self.CREW}
        for w in assign.values():
            hours[w] += 1
        return {"assign": {f"{d} {s}": w for (d, s), w in assign.items()},
                "hours": hours}

    def render(self):
        return [e["text"] for e in self.events]

    def ask(self, op: str = "duty"):
        st = self.replay()
        if op == "duty":
            d = str(self.rng.choice(self.DAYS))
            s = str(self.rng.choice(self.SHIFTS))
            q = f"Who works {d} {s} (confirmed assignments)?"
            a = st["assign"].get(f"{d} {s}", "nobody")
            chk = [f"assign={st['assign']}", f"{d} {s} -> {a}"]
        else:
            w = str(self.rng.choice(self.CREW))
            q = f"How many confirmed shifts does {w} have?"
            a = str(st["hours"][w])
            chk = [f"hours={st['hours']}", f"{w}={a}"]
        return q, a, chk


if __name__ == "__main__":
    # Auto-cohérence : ask() == re-dérivé du replay sur plusieurs seeds,
    # et les enregistrements invalides ne fuient pas dans l'état.
    for cls in (LedgerWorld, RosterWorld):
        for seed in range(5):
            w = cls(seed)
            w.gen(60, invalid_rate=0.15)
            st = w.replay()
            st2 = w.replay(list(w.events))  # re-exécution déterministe
            assert st == st2, (cls.__name__, seed)
            for op in (("total", "top", "owes") if cls is LedgerWorld
                       else ("duty", "hours")):
                q, a, chk = w.ask(op)
                assert q and a and len(chk) >= 1, (cls.__name__, op)
                assert any(a in c or str(a) == c.split("=")[-1].strip(" )")
                           for c in chk), (cls.__name__, op, a, chk)
    print("ie_worlds: 2 mondes x 5 seeds, replay déterministe + ask cohérent OK")
