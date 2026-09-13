"""
Tamagotchi : la mouche devient une créature à soigner.

Ses stats (satiété, humeur, énergie, propreté) décroissent avec le temps et
chaque action de soin est envoyée au connectome comme stimulus sensoriel :
l'intensité de la réponse motrice détermine la "réaction" de la mouche.
L'état persiste dans tamagotchi_state.json (décroissance hors-ligne incluse).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

STATE_PATH = Path(__file__).parent / "tamagotchi_state.json"

STATS = ("satiety", "happiness", "energy", "hygiene")
STAT_LABELS_FR = {
    "satiety": "Satiété",
    "happiness": "Humeur",
    "energy": "Énergie",
    "hygiene": "Propreté",
}

# Décroissance par heure (100 = parfait, 0 = au plus mal)
DECAY_PER_HOUR = {"satiety": 9.0, "happiness": 6.0, "energy": 7.0, "hygiene": 5.0}
OFFLINE_CAP_HOURS = 48.0
SLEEP_ENERGY_PER_HOUR = 30.0
MAX_STAT = 100.0

# Vocabulaire de stimuli : chaque action est une recette de ces canaux sensoriels
STIMULUS_CHANNELS = (
    "sweet", "touch", "smell", "water", "light", "vibration", "pattern", "frustration",
)
STIMULUS_RECIPES = {
    "feed":  {"sweet": 1.0, "smell": 0.8, "touch": 0.3},
    "pet":   {"touch": 1.0, "smell": 0.2},
    "clean": {"water": 1.0, "touch": 0.6},
    "wake":  {"light": 1.0, "vibration": 0.5},
    "chess": {"pattern": 1.0, "frustration": 0.4},
    # Comportements dimorphes (RAPPORT A4) : canal olfactif DA1/VA1v-like
    # pour la parade, mécanosensoriel répété pour la menace.
    "courtship": {"smell": 1.0, "vibration": 0.6, "touch": 0.3},
    "threat":    {"touch": 1.0, "vibration": 1.0, "frustration": 0.5},
}
STIMULUS_GAIN = 5.0

# Effets sur les stats par action (valeurs additives, clampées 0..100)
ACTION_EFFECTS = {
    "feed":  {"satiety": +30.0, "hygiene": -6.0, "happiness": +6.0},
    "pet":   {"happiness": +8.0, "energy": -1.0},
    "clean": {"hygiene": +100.0, "happiness": -4.0},
    "courtship": {"happiness": +10.0, "energy": -8.0},
    "threat":    {"happiness": -6.0, "energy": -5.0},
}

# Messages de réaction, par action et par intensité de la réponse neuronale
REACTION_MESSAGES = {
    "feed": {
        "strong": "Gloup ! Elle se jette sur le sirop, ailes vibrantes de plaisir 🍯",
        "weak":   "Elle goûte distraitement quelques gouttes…",
    },
    "pet": {
        "strong": "Elle frotte ses pattes avant, visiblement ravie d'être caressée 🥰",
        "weak":   "Un petit frémissement antennaire, elle te tolère.",
    },
    "clean": {
        "strong": "Pfuit ! Éclaboussée mais toute propre, elle s'essuie avec énergie 🚿",
        "weak":   "Elle survit au bain avec dignité.",
    },
    "wake": {
        "strong": "Elle s'étire, déploie ses ailes et bourdonne : prête ! ☀️",
        "weak":   "Elle ouvre un œil… grognonne… mais se lève.",
    },
    "chess": {
        "strong": "Ses lobes optiques s'affolent sur l'échiquier : elle ADORE ça ♟️",
        "weak":   "Elle regarde l'échiquier sans grande conviction.",
    },
    "courtship": {
        "strong": "Il déploie une aile et chante : parade nuptiale en cours 🎻🪰",
        "weak":   "Un petit frétillement d'aile, timide…",
    },
    "threat": {
        "strong": "Il fonce pattes en avant, ailes écartées : intimidation maximale 😠",
        "weak":   "Il fait un pas menaçant puis hésite.",
    },
}


class StimulusEncoder:
    """Projette une recette de stimulus (canaux nommés) sur les neurones sensoriels."""

    def __init__(self, n_sensory: int, seed: int = 7):
        self.n_sensory = n_sensory
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(len(STIMULUS_CHANNELS))
        self.proj = rng.normal(scale=scale, size=(len(STIMULUS_CHANNELS), n_sensory)).astype(np.float32)

    def encode(self, action: str, gain: float = STIMULUS_GAIN) -> torch.Tensor:
        recipe = STIMULUS_RECIPES[action]
        vec = np.zeros(len(STIMULUS_CHANNELS), dtype=np.float32)
        for i, channel in enumerate(STIMULUS_CHANNELS):
            vec[i] = recipe.get(channel, 0.0)
        return torch.tensor(vec @ self.proj * gain, dtype=torch.float32)


class FlyPet:
    """Le Tamagotchi : état, décroissance temporelle, actions, réactions neuronales."""

    def __init__(self, brain, n_sensory: int):
        self.brain = brain
        self.encoder = StimulusEncoder(n_sensory)
        self.state = self._load()
        self.apply_decay()

    # --- Persistance ---
    def _defaults(self) -> dict:
        now = time.time()
        return {
            "stats": {"satiety": 80.0, "happiness": 80.0, "energy": 80.0, "hygiene": 80.0},
            "xp": 0,
            "sleeping": False,
            "last_tick": now,
            "birth": now,
        }

    def _load(self) -> dict:
        state = self._defaults()
        if STATE_PATH.exists():
            try:
                saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                state["stats"].update(saved.get("stats", {}))
                for key in ("xp", "sleeping", "last_tick", "birth"):
                    if key in saved:
                        state[key] = saved[key]
                print(f"[pet] État chargé depuis {STATE_PATH}")
            except Exception as e:
                print(f"[pet] État illisible ({e}), nouvelle mouche.")
        return state

    def save(self):
        STATE_PATH.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # --- Temps qui passe ---
    def apply_decay(self):
        now = time.time()
        elapsed_h = min((now - self.state["last_tick"]) / 3600.0, OFFLINE_CAP_HOURS)
        if elapsed_h <= 0:
            self.state["last_tick"] = now
            return
        stats = self.state["stats"]
        if self.state["sleeping"]:
            stats["energy"] = min(MAX_STAT, stats["energy"] + SLEEP_ENERGY_PER_HOUR * elapsed_h)
            stats["satiety"] = max(0.0, stats["satiety"] - DECAY_PER_HOUR["satiety"] * 0.5 * elapsed_h)
            stats["happiness"] = max(0.0, stats["happiness"] - DECAY_PER_HOUR["happiness"] * 0.5 * elapsed_h)
            # Réveil automatique quand elle a dormi assez (ou 8h max)
            if stats["energy"] >= 99.0 or elapsed_h >= 8.0:
                self.state["sleeping"] = False
        else:
            for key, rate in DECAY_PER_HOUR.items():
                stats[key] = max(0.0, stats[key] - rate * elapsed_h)
            # Épuisée, elle s'endort toute seule
            if stats["energy"] <= 0.5:
                self.state["sleeping"] = True
        self.state["last_tick"] = now
        self.save()
        self.sync_neuromodulation()

    def sync_neuromodulation(self):
        """
        Octopamine : le bien-être de la mouche asservit le gain d'excitabilité
        global de son cerveau. Ravie → cerveau réactif ; épuisée → léthargique.
        """
        if self.brain is None:
            return
        if self.state["sleeping"]:
            self.brain.set_neuromod(0.45)
        else:
            self.brain.set_neuromod(0.6 + 0.8 * self.wellbeing())

    # --- Humeur ---
    def wellbeing(self) -> float:
        return float(np.mean([self.state["stats"][k] for k in STATS])) / MAX_STAT

    def mood(self) -> str:
        if self.state["sleeping"]:
            return "dort"
        stats = self.state["stats"]
        lowest = min(stats, key=stats.get)
        if stats[lowest] < 20.0:
            return {
                "satiety": "affamée",
                "happiness": "misérable",
                "energy": "épuisée",
                "hygiene": "inconfortable",
            }[lowest]
        if stats["happiness"] >= 75.0:
            return "ravie"
        if stats["happiness"] >= 50.0:
            return "contente"
        return "grognon"

    def mood_message(self) -> str:
        return {
            "dort": "Chut… elle dort profondément 💤",
            "ravie": "Elle bourdonne de bonheur autour de son terrarium 🪰✨",
            "contente": "Elle se lisse les antennes, l'air tranquille.",
            "grognon": "Elle boude dans un coin du terrarium 😤",
            "affamée": "Elle tourne en rond : elle crève de faim ! 🍯",
            "misérable": "Elle a l'air vraiment triste…",
            "épuisée": "Ses ailes traînent : elle est épuisée 💤",
            "inconfortable": "Elle se démange : une petite toilette s'impose 🚿",
        }[self.mood()]

    def chess_noise(self) -> float:
        """Bruit ajouté aux scores des coups : une mouche mal en point joue mal."""
        return (1.0 - self.wellbeing()) * 0.4

    # --- Réaction neuronale à une action ---
    @torch.no_grad()
    def react(self, action: str) -> dict:
        currents = self.encoder.encode(action).to(self.brain.device)
        result = self.brain.run(currents, n_steps=200, record_every=10)
        motor_mean = float(result["motor_mean"].mean().item())
        strength = float(np.tanh(motor_mean * 300.0))  # 0..1
        bucket = "strong" if strength >= 0.35 else "weak"
        return {
            "strength": round(strength, 3),
            "message": REACTION_MESSAGES[action][bucket],
            "frames": [],
        }

    # --- Actions de soin ---
    def do_action(self, action: str) -> dict:
        self.apply_decay()
        if action not in ("feed", "pet", "clean", "wake", "sleep", "courtship", "threat"):
            raise ValueError(f"Action inconnue : {action}")

        if action == "sleep":
            if self.state["sleeping"]:
                return {"message": "Elle dort déjà ! 💤", "frames": [], "applied": False}
            self.state["sleeping"] = True
            self.state["xp"] += 2
            self.save()
            return {"message": "Les lumières s'éteignent… bonne nuit 🌙", "frames": [], "applied": True}

        if action == "wake":
            if not self.state["sleeping"]:
                return {"message": "Elle est déjà réveillée !", "frames": [], "applied": False}
            self.state["sleeping"] = False
            self.state["xp"] += 2
            self.save()
            return {**self.react("wake"), "applied": True}

        if self.state["sleeping"]:
            return {"message": "Chut ! Elle dort. Réveille-la d'abord 💤", "frames": [], "applied": False}

        # Effets sur les stats (avec un refus si elle n'a plus faim)
        if action == "feed" and self.state["stats"]["satiety"] > 90.0:
            self.state["stats"]["happiness"] = min(MAX_STAT, self.state["stats"]["happiness"] - 5.0)
            self.state["xp"] += 1
            self.save()
            return {"message": "Elle n'a plus faim du tout et boude le sirop 🙄", "frames": [], "applied": False}

        for key, delta in ACTION_EFFECTS[action].items():
            self.state["stats"][key] = float(
                np.clip(self.state["stats"][key] + delta, 0.0, MAX_STAT)
            )
        self.state["xp"] += 5
        self.save()
        return {**self.react(action), "applied": True}

    # --- Mini-jeu échecs ---
    def on_chess_move(self, motor: float | None = None, fast: bool = False) -> dict:
        """Appelé après chaque coup de la mouche : ça la stimule mais la fatigue.

        motor : activité motrice moyenne du coup gagnant (déjà simulée par le
            moteur) → réaction SANS simu (+73 s économisées sur MaleCNS).
            fast=True (parties 2 humains, pas de simu moteur) → réaction texte.
        Sans les deux : repli historique (simu stimulus "chess", 200 pas)."""
        self.apply_decay()
        if self.state["sleeping"]:
            return {"message": "💤", "frames": [], "applied": False}
        self.state["stats"]["energy"] = max(0.0, self.state["stats"]["energy"] - 3.0)
        self.state["stats"]["satiety"] = max(0.0, self.state["stats"]["satiety"] - 2.0)
        self.state["stats"]["happiness"] = min(MAX_STAT, self.state["stats"]["happiness"] + 2.0)
        self.state["xp"] += 3
        self.save()
        if fast:
            return {"message": "Elle suit la partie d'un œil vif, ailes frémissantes ♟️",
                    "strength": 0.2, "frames": [], "applied": True}
        if motor is not None:
            strength = float(np.tanh(motor * 300.0))  # même échelle que react()
            bucket = "strong" if strength >= 0.35 else "weak"
            return {"strength": round(strength, 3),
                    "message": REACTION_MESSAGES["chess"][bucket],
                    "frames": [], "applied": True}
        return {**self.react("chess"), "applied": True}

    # --- Sérialisation pour l'API ---
    def status(self) -> dict:
        self.apply_decay()
        level = self.state["xp"] // 100 + 1
        return {
            "stats": self.state["stats"],
            "stat_labels": STAT_LABELS_FR,
            "sleeping": self.state["sleeping"],
            "mood": self.mood(),
            "mood_message": self.mood_message(),
            "wellbeing": round(self.wellbeing(), 3),
            "chess_noise": round(self.chess_noise(), 3),
            "octopamine": round(self.brain.neuromod, 3) if self.brain else 1.0,
            "level": int(level),
            "xp": self.state["xp"],
            "xp_next": int(level * 100),
            "age_hours": round((time.time() - self.state["birth"]) / 3600.0, 1),
        }
