"""
typesafe_judge — client minimal de l'API TypeSafe System One (Jev).

Des jugements typés (Choice / Noul / Score) comme primitives de code :
le code possède le workflow, le modèle fournit le sens commun sémantique.

Clé : variable d'environnement TYPESAFE_API_KEY (jamais dans le code/git).
Zéro dépendance (urllib stdlib). Docs : https://docs.typesafe.ai/api
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class TypeSafeError(RuntimeError):
    pass


def evaluate(state, questions: dict, model: str = MODEL,
             timeout: int = 60, retries: int = 2) -> dict:
    """Évalue `state` contre `questions` ({id: {type, instructions, criteria?}}).
    Retourne le dict réponse brut (model, answers, usage). Lève TypeSafeError
    si clé absente, HTTP non-retryable, ou échec après retries (429/529 backoff)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise TypeSafeError("TYPESAFE_API_KEY manquant (variable d'environnement).")
    body = json.dumps({"state": state, "model": model,
                       "questions": questions}).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = TypeSafeError(f"TypeSafe HTTP {e.code}: "
                                 f"{e.read().decode()[:300]}")
            if e.code not in (429, 529):
                raise last
        except Exception as e:  # réseau/timeout : retry puis abandon
            last = TypeSafeError(f"TypeSafe réseau: {e}")
        time.sleep(2 ** attempt)
    raise last  # type: ignore[misc]


def care_questions() -> dict:
    """Questions de triage Tamagotchi (pures, testables sans réseau)."""
    return {
        "care_action": {
            "type": "choice",
            "instructions": "Which single care action does the fly need most?",
            "criteria": {"feed": "hungry or low satiety",
                         "pet": "low happiness, needs comfort",
                         "clean": "low hygiene, dirty",
                         "sleep": "low energy, exhausted, or already asleep"},
        },
        "is_critical": {
            "type": "noul",
            "instructions": "Does the fly need care right now (not later today)?",
            "criteria": {
                "true": "any stat at or near zero (<=15/100), or several stats low at once",
                "false": "all stats comfortably above 15, routine care suffices",
            },
        },
        "mood": {
            "type": "score",
            "instructions": "How good is the fly's mood?",
            "criteria": ["miserable", "okay", "happy"],
        },
    }


def advise_care(stats: dict, sleeping: bool, timeout: int = 15) -> dict:
    """Triage de l'état Tamagotchi (1 appel batché). stats: {satiety,
    happiness, energy, hygiene}. Lève TypeSafeError sans clé/réseau."""
    return evaluate({"stats": stats, "sleeping": bool(sleeping)},
                    care_questions(), timeout=timeout)


if __name__ == "__main__":
    # Démo opt-in (1 seul appel, 3 questions indépendantes batchées) :
    # triage de l'état Tamagotchi -> action de soin.
    demo_state = {"satiety": 0.15, "mood": 0.3, "energy": 0.6,
                  "cleanliness": 0.8, "asleep": False,
                  "note": "hasn't eaten in 6 hours, keeps missing the food"}
    res = advise_care({k: demo_state.get(k, 0.5) for k in
                       ("satiety", "happiness", "energy", "hygiene")},
                      demo_state.get("asleep", False))
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------------------
# Briques projet (FlyChess/FlyIntel) : états réels -> questions batchées (1 appel).
# Sans clé, evaluate() lève TypeSafeError : les appelants doivent catcher et
# passer leur chemin (jamais de réseau dans les chemins chauds/tests).
# ---------------------------------------------------------------------------

def judge_bench_entry(entry: dict) -> dict:
    """Santé qualitative d'une entrée leaderboard (1 appel batché).
    entry: {tag, n_neurons, n_synapses, domains: {nom: {score, detail}}}."""
    state = {"tag": entry.get("tag"), "n_neurons": entry.get("n_neurons"),
             "n_synapses": entry.get("n_synapses"),
             "domains": entry.get("domains", {})}
    return evaluate(state, {
        "overall_health": {
            "type": "choice",
            "instructions": "Given these benchmark domains of a simulated fly-brain connectome, what is the overall health of this run?",
            "criteria": {
                "healthy": "key domains report plausible values, no errors",
                "degraded": "works but something is off: silent network, missing optional readout, suspicious score",
                "broken": "errors, dead simulation despite drive, or degenerate outputs everywhere",
            },
        },
        "needs_attention": {
            "type": "noul",
            "instructions": "Does any domain need human attention before trusting this leaderboard entry?",
        },
    })


def triage_notes(notes: list[dict], max_notes: int = 5) -> dict:
    """Trie des notes apprises {title, source, excerpt} : domaine + garder ?
    1 appel batché (2 questions/note). Au-delà de max_notes, tronque."""
    notes = notes[:max_notes]
    state = [{"title": n.get("title"), "source": n.get("source"),
              "excerpt": (n.get("excerpt") or "")[:600]} for n in notes]
    qs: dict = {}
    for i, n in enumerate(state):
        # Le sens complet dans la question (titre cité) : le modèle ne peut
        # pas se tromper d'index dans une liste d'états.
        ref = f"the note titled '{n.get('title')}'"
        qs[f"domain_{i}"] = {
            "type": "choice",
            "instructions": f"What domain is {ref} about?",
            "criteria": {"chess": "chess, openings, endgames, engine evaluation",
                         "neuro": "neurons, connectome, Drosophila, plasticity, reflexes",
                         "tech": "kernels, SIMD, quantization, accelerators",
                         "other": "none of the above"},
        }
        qs[f"keep_{i}"] = {
            "type": "noul",
            "instructions": f"Is {ref} worth keeping in a fly-brain research knowledge base?",
        }
    return evaluate(state, qs)


def rank_options(topic: str, options: dict[str, str], evidence: str = "") -> dict:
    """Arbitre entre options d'architecture (1 appel) : retourne le choix +
    la distribution. `evidence` = faits mesurés, pas des opinions."""
    return evaluate({"topic": topic, "measured_evidence": evidence}, {
        "best_option": {
            "type": "choice",
            "instructions": f"Given the measured evidence, which option best serves: {topic}?",
            "criteria": options,
        },
    })


# ---------------------------------------------------------------------------
# Phases 1-3 : briques produit. Règles : 1 appel batché par fonction, jamais
# dans les chemins chauds, seuils calibrés ci-dessous (phase 1, voir rapport).
# ---------------------------------------------------------------------------

TS_KEEP_THRESHOLD = 0.5        # note gardée si keep >= seuil (calibré : Jev ne garde
                               # que le spécifique fly-brain, le générique échecs/tech est droppé)
TS_ESCALATE_CONFIDENCE = 0.6   # sous ce seuil -> vérificateur cher (stockfish depth 12)


def commentate_move(fen: str, move_uci: str, cp_before: float, cp_after: float,
                    timeout: int = 20) -> dict:
    """Commentaire d'un coup (1 appel batché). Ajoute needs_escalation
    (confiance < TS_ESCALATE_CONFIDENCE) et un texte FR templatisé."""
    res = evaluate({"fen": fen, "move": move_uci, "cp_before": cp_before,
                    "cp_after": cp_after}, {
        "is_blunder": {
            "type": "noul",
            "instructions": "Does this move blunder (cp drop of roughly 150 or more)?",
        },
        "sharpness": {
            "type": "score",
            "instructions": "How sharp is the resulting position?",
            "criteria": ["calm", "tense", "wild"],
        },
        "style": {
            "type": "choice",
            "instructions": "What style is this move?",
            "criteria": {"tactical": "sacrifice, capture, check, forcing sequence",
                         "positional": "quiet improvement, prophylaxis, maneuvering",
                         "defensive": "parries a threat, consolidates, retreats"},
        },
    }, timeout=timeout)
    a = res["answers"]
    res["needs_escalation"] = min(a["sharpness"]["confidence"],
                                  a["style"]["confidence"]) < TS_ESCALATE_CONFIDENCE
    res["commentary_fr"] = format_commentary(a)
    return res


def format_commentary(a: dict) -> str:
    """Template FR pur (zéro réseau) depuis les jugements d'un coup."""
    bl = a["is_blunder"]["noul"]
    sh = a["sharpness"]["score"]
    st = a["style"]["choice"]
    bits = ["Gaffe (%.2f)." % bl if bl > 0.5 else "Coup propre (%.2f)." % (1 - bl),
            "Position " + ("calme." if sh < 0.7 else "tendue." if sh < 1.4 else "sauvage."),
            {"tactical": "Esprit tactique.",
             "positional": "Esprit positionnel.",
             "defensive": "D'abord parer."}[st]]
    return " ".join(bits)


def route_query(query: str, n_notes: int, timeout: int = 15) -> dict:
    """Répondre depuis knowledge ou chercher le web ? (1 appel)."""
    return evaluate({"query": query, "notes_available": n_notes}, {
        "source": {
            "type": "choice",
            "instructions": "Where should this question be answered from?",
            "criteria": {"knowledge": f"the {n_notes} learned notes likely cover it",
                         "web": "needs fresh or external information"},
        },
    })


def detect_theater_event(stats: dict, timeout: int = 15) -> dict:
    """Le flux live vaut-il une réaction caméra/lumière ? stats: {spike_hz,
    motor_hz, burst_peak, silent_s}. Downsampler côté appelant (1/10 s max)."""
    return evaluate(stats, {
        "interesting": {
            "type": "noul",
            "instructions": "Is something worth reacting to happening in this brain-activity snapshot?",
        },
        "event": {
            "type": "choice",
            "instructions": "What kind of moment is this?",
            "criteria": {"burst": "sudden spike surge",
                         "silence": "activity died out",
                         "rally": "sustained motor recruitment",
                         "nothing": "business as usual"},
        },
    })


def route_intent(text: str, timeout: int = 15) -> dict:
    """Porte d'entrée langage naturel v1 : handler + confiance (1 appel).
    L'exécution reste côté appelant (pas d'effet de bord ici)."""
    return evaluate({"request": text}, {
        "handler": {
            "type": "choice",
            "instructions": "Which subsystem should handle this request?",
            "criteria": {"pet": "feed, pet, clean, sleep, wake, mood, care actions",
                         "chess": "play, move, board, game, position",
                         "theater": "watch, show, brain, live stream, display",
                         "lab": "lesion, heal, experiment, reflex test",
                         "learn": "search, learn, remember, knowledge questions"},
        },
    })


def personality_scores(summary: str, timeout: int = 15) -> dict:
    """Personnalité depuis un résumé d'historique (1 appel, 2 scores).
    L'appelant mappe vers température/gains (jamais ici)."""
    return evaluate({"recent_history": summary}, {
        "boldness": {
            "type": "score",
            "instructions": "How bold is the fly behaving?",
            "criteria": ["timid", "balanced", "reckless"],
        },
        "sociability": {
            "type": "score",
            "instructions": "How much does the fly seek interaction?",
            "criteria": ["withdrawn", "neutral", "attention-seeking"],
        },
    })


def check_citations(pairs: list[tuple[str, str]], timeout: int = 20) -> dict:
    """Chaque affirmation est-elle soutenue par son extrait ? (1 appel batché,
    1 question Noul par paire ; sens complet dans la question)."""
    qs = {}
    for i, (claim, excerpt) in enumerate(pairs):
        qs[f"supported_{i}"] = {
            "type": "noul",
            "instructions": f"Is this claim supported by the excerpt? Claim: '{claim}'",
            "criteria": {"true": "excerpt states or directly implies it",
                         "false": "not present or contradicted"},
        }
    return evaluate([{"excerpt": ex[:600]} for _, ex in pairs], qs)
