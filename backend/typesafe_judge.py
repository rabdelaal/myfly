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
            "instructions": "Is the fly in a critical state needing immediate care?",
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
                "healthy": "key domains report plausible values, no errors, no silent brain",
                "degraded": "works but a domain looks off (silent network, degenerate readout, suspicious score)",
                "broken": "errors, missing readouts, or numbers that indicate a dead/misconfigured simulation",
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
