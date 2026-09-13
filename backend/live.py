"""
Cerveau en direct : simulation cérébrale CONTINUE de la mouche, diffusée en
temps réel par WebSocket.

- Un FlyBrain persistant et dédié (l'état neural est conservé entre les
  frames : ce cerveau « vit », il n'est pas réinitialisé à chaque question).
- Bruit sensoriel spontané : la mouche sent son terrarium en permanence,
  avec une vivacité modulée par son humeur (octopamine du Tamagotchi).
- Les actions de soin (nourrir, caresser, laver…) et les coups d'échecs sont
  injectés comme stimuli sensoriels visibles : on VOIT la réaction cérébrale.
- Ne tourne que lorsqu'au moins un client WebSocket est abonné (zéro CPU sinon).
"""
import asyncio
import time

import numpy as np
import torch

from brain import FlyBrain
from tamagotchi import StimulusEncoder

FRAME_INTERVAL = 1.0 / 30.0   # 30 fps
STEPS_PER_FRAME = 5           # 5 ms simulées par frame → ralenti ≈ 6.6× (esthétique)
DT = 1.0                      # ms
MAX_SPIKES_PER_FRAME = 800
NOISE_SIGMA_AWAKE_CURRENT = 4.5   # mode current : σ élevé nécessaire
NOISE_SIGMA_AWAKE_ALPHA = 0.3     # mode alpha (Shiu) : le réseau amplifie ×37
NOISE_SIGMA_SLEEP = 0.05
STIM_STEPS = 30               # 30 ms de stimulus par événement


class LiveSim:
    def __init__(self, conn, pet, brain_kwargs: dict | None = None):
        self.brain = FlyBrain(
            conn["W"], conn["is_sensory"], conn["is_motor"], conn["is_descending"],
            **(brain_kwargs or {}),
        )
        if conn.get("sugar_grn_idx") is not None and len(conn["sugar_grn_idx"]):
            self.brain.set_sugar_neurons(conn["sugar_grn_idx"])
        self.encoder = StimulusEncoder(int(conn["is_sensory"].sum()))
        # σ du bruit ambiant selon le modèle de synapse
        if getattr(self.brain, "synapse_model", "current") == "alpha":
            self.noise_sigma_awake = NOISE_SIGMA_AWAKE_ALPHA
        else:
            self.noise_sigma_awake = NOISE_SIGMA_AWAKE_CURRENT
        self.pet = pet
        self.n_sensory = self.brain.n_sensory
        self.clients: set = set()
        self.task: asyncio.Task | None = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.pending_stim = None
        self.stim_steps_left = 0
        self.caption = None
        self._frame_count = 0
        # Pas par frame adaptatif : mesurer le coût réel d'un pas à cette échelle
        self.brain.reset(1)
        _t0 = time.perf_counter()
        for _ in range(2):
            self.brain.step(None)
        t_step = (time.perf_counter() - _t0) / 2
        self.frame_interval = FRAME_INTERVAL if t_step < 0.05 else 1.0 / 10.0
        self.steps_per_frame = max(1, min(STEPS_PER_FRAME, int(0.020 / max(t_step, 1e-4))))
        print(f"[live] pas LIF mesuré : {t_step*1000:.1f} ms -> "
              f"{self.steps_per_frame} pas/frame @ {1/self.frame_interval:.0f} fps")
        # Masques population pré-convertis en tenseurs (évite 20 allocs/frame)
        _hot = np.asarray(conn.get("is_hotspot", np.zeros(len(conn["is_sensory"]), bool)), dtype=bool)
        _fru = np.asarray(conn.get("is_fru", np.zeros(len(conn["is_sensory"]), bool)), dtype=bool)
        self._tmasks = {
            "sensory": torch.tensor(conn["is_sensory"], device=self.brain.device),
            "inter": torch.tensor(
                ~(conn["is_sensory"] | conn["is_motor"] | conn["is_descending"]),
                device=self.brain.device),
            "motor": torch.tensor(conn["is_motor"], device=self.brain.device),
            "descending": torch.tensor(conn["is_descending"], device=self.brain.device),
            # Populations dimorphes du papier Cell (A4) : grappes fru/dsx
            "hotspot": torch.tensor(_hot, device=self.brain.device),
            "fru": torch.tensor(_fru, device=self.brain.device),
        }
        self._sizes = {k: max(int(m.sum()), 1) for k, m in self._tmasks.items()}

    # --- Gestion des abonnés ---
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.clients.add(q)
        self._ensure_task()
        return q

    def unsubscribe(self, q) -> None:
        self.clients.discard(q)
        if not self.clients and self.task is not None:
            self.task.cancel()
            self.task = None

    def _ensure_task(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self._run())

    def event(self, action: str, caption: str) -> None:
        """Injecte un stimulus (feed/pet/clean/wake/chess/courtship/threat) avec sa légende."""
        if self.clients:
            self.events.put_nowait((action, caption))

    def _sugar_available(self) -> bool:
        si = getattr(self.brain, "_sugar_idx", None)
        return si is not None and len(si) > 0

    # --- Boucle de simulation ---
    async def _run(self) -> None:
        noise_rng = np.random.default_rng(42)
        print("[live] Cerveau en direct démarré "
              f"({len(self.clients)} spectateur(s))")
        try:
            while self.clients:
                t0 = time.perf_counter()

                # 1. Événements de soin / jeu en attente
                caption_now = None
                while not self.events.empty():
                    action, caption = self.events.get_nowait()
                    if action == "lab":
                        # ⚗️ Labo : légende seule (l'état du cerveau est déjà
                        # modifié par la lésion, pas de stimulus à injecter)
                        self.pending_stim = None
                        self.stim_steps_left = 0
                    elif (action == "feed" and self._sugar_available()):
                        # Réflexe biologique : Poisson 100 Hz sur les vraies GRN sucrées
                        self.brain.set_sugar_drive(100.0, 250)
                        self.pending_stim = None
                        self.stim_steps_left = 0
                    else:
                        self.pending_stim = self.encoder.encode(action)
                        self.stim_steps_left = STIM_STEPS
                    caption_now = caption

                # 2. Courant externe de ce frame : événement > bruit spontané
                if self.stim_steps_left > 0 and self.pending_stim is not None:
                    I_ext = self.pending_stim
                    self.stim_steps_left -= 1
                else:
                    sigma = (NOISE_SIGMA_SLEEP if self.pet.state["sleeping"]
                             else self.noise_sigma_awake * (0.4 + 0.6 * self.pet.wellbeing()))
                    I_ext = torch.randn(self.n_sensory) * sigma

                # 3. L'humeur pilote l'excitabilité (octopamine partagée)
                self.brain.set_neuromod(self.pet.brain.neuromod)

                # 4. Simulation + comptage par population
                counts = {k: 0.0 for k in self._tmasks}
                for _ in range(self.steps_per_frame):
                    self.brain.step(I_ext)
                    sp = self.brain.spikes[:, 0]
                    for k, tmask in self._tmasks.items():
                        counts[k] += float(sp[tmask].sum().item())

                # 5. Frame : indices des spikes (colonne 0), capped
                idx = torch.nonzero(self.brain.spikes[:, 0], as_tuple=False).squeeze(1)
                if idx.numel() > MAX_SPIKES_PER_FRAME:
                    idx = idx[:MAX_SPIKES_PER_FRAME]
                sim_ms = self.steps_per_frame * DT
                payload = {
                    "type": "live",
                    "spikes": idx.cpu().tolist(),
                    "hz": {k: round(v / self._sizes[k] / (sim_ms / 1000.0), 1)
                           for k, v in counts.items()},
                    "mood": self.pet.mood(),
                    "sleeping": self.pet.state["sleeping"],
                    "caption": caption_now,
                    "neuromod": round(self.brain.neuromod, 3),
                }

                # 6. Diffusion (les clients lents perdent des frames, c'est voulu)
                for q in list(self.clients):
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass

                # 7. Décroissance des stats Tamagotchi ~1×/10 s
                self._frame_count += 1
                if self._frame_count % 300 == 0:
                    self.pet.apply_decay()

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, self.frame_interval - elapsed))
        except asyncio.CancelledError:
            pass
        finally:
            print("[live] Cerveau en direct arrêté")
