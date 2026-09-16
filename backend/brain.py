"""
Simulation LIF (Leaky Integrate-and-Fire) du connectome sur GPU/CPU.
Vectorisée avec PyTorch sparse : les états sont batchés en (n_neurones, B),
ce qui permet de simuler B configurations (ex. 20 coups candidats) en parallèle
avec UNE seule multiplication sparse par pas de temps.

Neuromodulation : `neuromod` simule le taux d'octopamine (l'hormone d'excitation
des insectes) — un gain global appliqué au courant synaptique. Le Tamagotchi
l'asservit au bien-être de la mouche.
"""
import warnings

import torch
import numpy as np
import scipy.sparse as sp

# Format CSR en beta côté PyTorch : warning inutile à chaque import
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")

# Cache partagé : plusieurs FlyBrain (moteur, live) sur le même connectome
# ne doivent construire le tenseur sparse qu'une seule fois (~1 Go sur MaleCNS).
_W_TORCH_CACHE = {}

# Au-delà de ce nnz, le kernel C mono-thread perd contre torch multi-thread
NATIVE_MAX_NNZ = 20_000_000


class FlyBrain:
    """
    Simulateur LIF batché.

    Paramètres biologiques (drosophile) :
      - tau_m = 20 ms (constante de membrane)
      - V_th = 1.0 (seuil de décharge, unités normalisées)
      - V_reset = 0.0
      - dt = 1 ms (1 kHz)
      - neuromod = gain octopaminergique global (1.0 = neutre)
    """

    def __init__(
        self,
        W: sp.csr_matrix,
        is_sensory: np.ndarray,
        is_motor: np.ndarray,
        is_descending: np.ndarray,
        tau_m: float = 20.0,
        V_th: float = 1.0,
        V_reset: float = 0.0,
        dt: float = 1.0,
        device: str | None = None,
        synapse_model: str = "current",
        weight_scale: float = 1.0,
        V_rest: float | None = None,
        tau_syn: float = 5.0,
        delay_steps: int = 2,
        refractory_steps: int = 3,
        stim_p_per_mv: float = 0.01,
    ):
        """
        synapse_model :
          - "current" : I_syn = gain × (W @ spikes) — modèle historique
          - "alpha"   : modèle de Shiu et al. (Nature 2024) — α-synapses :
            g <- g·exp(-dt/τ_syn) + W@spikes retardés ; I_syn = g ;
            réfractaire absolu ; V_rest/V_th en mV (−52/−45).
            weight_scale multiplie les poids (ex. : brut × Wsyn).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.synapse_model = synapse_model
        self.weight_scale = float(weight_scale)
        self.n = W.shape[0]
        self.n_synapses = int(W.nnz)
        self.tau_m = tau_m
        self.V_th = V_th
        self.V_reset = V_reset
        self.V_rest = V_rest if V_rest is not None else 0.0
        self.dt = dt
        self.tau_syn = tau_syn
        self.delay_steps = max(1, int(delay_steps))
        self.refractory_steps = int(refractory_steps)
        # Codage en fréquence des stimuli (mode alpha) : I_ext positif est
        # converti en proba de spike Poisson (p = I × stim_p_per_mv par pas),
        # comme la stimulation Poisson de Shiu et al.
        self.stim_p_per_mv = float(stim_p_per_mv)
        self.neuromod = 1.0
        # Drive Poisson des GRN sucrées (réflexe d'alimentation, Shiu et al.)
        self._sugar_idx = None
        self._sugar_drive_left = 0
        self._sugar_hz = 0.0
        # Lésion virtuelle (⚗️ Labo) : ces neurones ne tirent JAMAIS
        self._lesion_idx = None

        # Format CSR : ~16x plus rapide que COO pour sparse.mm côté CPU,
        # et tout aussi supporté par torch.sparse.mm côté GPU.
        cache_key = (id(W), self.weight_scale)
        cached = _W_TORCH_CACHE.get(cache_key)
        if cached is not None:
            self.W = cached
        else:
            W_coo = W.tocoo()
            indices = torch.tensor(
                np.vstack([W_coo.row, W_coo.col]), dtype=torch.long
            )
            values = torch.tensor(W_coo.data * self.weight_scale, dtype=torch.float32)
            self.W = torch.sparse_coo_tensor(
                indices, values, size=(self.n, self.n), device=self.device
            ).coalesce().to_sparse_csr()
            _W_TORCH_CACHE[cache_key] = self.W

        # Masques
        self.is_sensory = torch.tensor(is_sensory, dtype=torch.bool, device=self.device)
        self.is_motor = torch.tensor(is_motor, dtype=torch.bool, device=self.device)
        self.is_descending = torch.tensor(
            is_descending, dtype=torch.bool, device=self.device
        )
        # Indices des neurones sensoriels, pour disperser le courant d'entrée
        self._sensory_idx = torch.where(self.is_sensory)[0]
        self.n_sensory = int(self._sensory_idx.numel())

        # Backend natif optionnel (kernel C vérifié bit-exact, CPU uniquement) :
        # même sémantique que la version torch, sans les overheads d'opérateurs.
        # Désactivé en mode alpha (conductances non implémentées en C) et sur
        # les très gros connectomes (mono-thread perd vs torch).
        self._native = None
        if (self.device == "cpu" and self.n_synapses <= NATIVE_MAX_NNZ
                and synapse_model == "current"):
            try:
                from native_lif import NativeLIF
                self._native = NativeLIF()
                self._indptr = np.ascontiguousarray(W.indptr, dtype=np.int32)
                self._indices = np.ascontiguousarray(W.indices, dtype=np.int32)
                self._data = np.ascontiguousarray(W.data, dtype=np.float32)
                self._sensory_np = np.asarray(is_sensory)
                self._isyn_bufs = {}
            except Exception as e:
                print(f"[brain] backend natif indisponible ({e}), repli torch.")
                self._native = None

        # Backend natif α event-driven (lif_alpha_ed.dll) : layout CSC, ne
        # disperse que les neurones actifs (~7 %) au lieu du sparse.mm dense.
        # Uniquement batch=1 (prod live.py), CPU. Gagne sur les gros
        # connectomes où le sparse.mm torch ~51-71 ms/step est memory-bound.
        self._native_alpha = None
        self._native_alpha16 = None
        if (self.device == "cpu" and synapse_model == "alpha"):
            try:
                from native_lif_alpha import NativeLIFAlpha, csc_from_csr
                self._native_alpha = NativeLIFAlpha()
                self._csc_colptr, self._csc_row, self._csc_data = \
                    csc_from_csr(W, self.n)
                # Le path torch multiplie les poids par weight_scale : le CSC
                # natif doit porter la même échelle (sinon réseau ~16× trop
                # faible, silencieux). ponytail: pas de variante int16 ici —
                # W est normalisé (flottants ~0.05), round() en int16 zérote
                # 99 % des synapses ; réactiver seulement sur poids bruts entiers.
                if self.weight_scale != 1.0:
                    self._csc_data = (self._csc_data * np.float32(self.weight_scale)).astype(np.float32)
                self._csc_nnz = int(self._csc_data.shape[0])
                self._csc16_data = None
            except Exception as e:
                print(f"[brain] backend natif α indisponible ({e}), repli torch.")
                self._native_alpha = None

        self.reset()

    def set_neuromod(self, gain: float):
        """Asservit le gain octopaminergique global (ex. au bien-être du Tamagotchi)."""
        self.neuromod = float(gain)

    def set_sugar_neurons(self, idx: np.ndarray):
        """Déclare les GRN sucrées (indices) pour le drive Poisson du réflexe
        d'alimentation (Shiu et al. : 100 Hz → MN9)."""
        self._sugar_idx = torch.tensor(np.asarray(idx, dtype=np.int64),
                                       device=self.device)

    def set_sugar_drive(self, hz: float, steps: int):
        """Active le drive Poisson (hz) sur les GRN sucrées pendant steps ms."""
        if self._sugar_idx is None or len(self._sugar_idx) == 0:
            return
        self._sugar_hz = float(hz)
        self._sugar_drive_left = int(steps)

    def set_lesion(self, indices) -> None:
        """
        Lésion virtuelle (⚗️ Labo) : les neurones d'indices donnés ne tirent
        JAMAIS — leurs spikes sont forcés à zéro après chaque pas (ni
        récurrence, ni transmission, ni drive). indices vide/None → efface.
        """
        if indices is None or len(indices) == 0:
            self._lesion_idx = None
            return
        self._lesion_idx = torch.tensor(np.asarray(indices, dtype=np.int64),
                                        device=self.device)
        # Le kernel C natif ne connaît pas la lésion → repli torch (correct,
        # plus lent). Sans impact en prod MaleCNS (26M synapses > NATIVE_MAX).
        self._native = None

    def clear_lesion(self) -> None:
        """Efface la lésion (la mouche guérit)."""
        self._lesion_idx = None

    def reset(self, batch_size: int = 1):
        self.V = torch.full((self.n, batch_size), self.V_rest, device=self.device)
        self.spikes = torch.zeros(self.n, batch_size, device=self.device)
        self.batch_size = batch_size
        self.t = 0
        if self.synapse_model == "alpha":
            self.g = torch.zeros(self.n, batch_size, device=self.device)
            self._refr = torch.zeros(self.n, batch_size, dtype=torch.int32,
                                     device=self.device)
            self._delay = [torch.zeros(self.n, batch_size, device=self.device)
                           for _ in range(self.delay_steps)]
            self._syn_decay = float(np.exp(-self.dt / self.tau_syn))

    def _prepare_I_native(self, I_ext: torch.Tensor | None):
        """Prépare I_full (n, B) numpy pour le kernel C, ou None."""
        if I_ext is None:
            return None
        I_ext = I_ext.detach().cpu().numpy()
        if I_ext.ndim == 1:
            I_ext = I_ext[:, None]
        # Seuls les neurones sensoriels reçoivent le courant externe
        padded = np.zeros((self.n, I_ext.shape[1]), dtype=np.float32)
        if I_ext.shape[0] == self.n:
            padded[self._sensory_np] = I_ext[self._sensory_np]
        else:
            padded[self._sensory_np] = I_ext[: self.n_sensory]
        return padded

    def _prepare_I_torch(self, I_ext: torch.Tensor | None):
        """Prépare I_full (n, B) tensor (courants masqués aux sensoriels)."""
        if I_ext is None:
            return None
        I_ext = I_ext.to(self.device)
        if I_ext.dim() == 1:
            I_ext = I_ext.unsqueeze(1)
        masked = torch.zeros(self.n, I_ext.shape[1], device=self.device)
        if I_ext.shape[0] == self.n:
            masked[self.is_sensory] = I_ext[self.is_sensory]
        else:
            masked[self._sensory_idx, :] = I_ext[: self.n_sensory, :]
        return masked

    @torch.no_grad()
    def _step_prepared(self, I_full) -> torch.Tensor:
        """Un pas avec le courant externe DÉJÀ préparé (n, B) ou None."""
        B = self.spikes.shape[1]

        if self.synapse_model == "alpha":
            # --- Modèle Shiu et al. (Nature 2024) : α-synapses 3 équations ---
            # 1) délai spike-à-effet (1,8 ms → buffer circulaire). Pas de clone :
            # self.spikes est re-lié (nouveau tenseur) à chaque pas, l'objet
            # stocké n'est donc jamais muté par les pas suivants (les edits
            # sugar/lésion touchent le nouveau tenseur, cf. lignes ci-dessous).
            delayed = self._delay.pop(0)
            self._delay.append(self.spikes)

            # --- Backend natif α event-driven (batch=1, CPU) ---
            if (self._native_alpha is not None and B == 1):
                V_np = self.V.numpy()
                g_np = self.g.numpy()
                refr_np = self._refr.numpy()
                delayed_np = delayed.numpy()
                spikes_out = np.zeros(self.n, dtype=np.float32)
                if self._native_alpha16 is not None:
                    # Variante int16 : 2× moins de trafic mémoire, bit-exact
                    self._native_alpha16.step16(
                        self._csc_colptr, self._csc_row, self._csc16_data,
                        delayed_np, V_np, g_np, refr_np,
                        self.V_rest, self.V_th, self.V_reset, self.neuromod,
                        self.tau_m, self.dt, self._syn_decay, self.refractory_steps,
                        spikes_out,
                    )
                else:
                    self._native_alpha.step(
                        self._csc_colptr, self._csc_row, self._csc_data,
                        delayed_np, V_np, g_np, refr_np,
                        self.V_rest, self.V_th, self.V_reset, self.neuromod,
                        self.tau_m, self.dt, self._syn_decay, self.refractory_steps,
                        spikes_out,
                    )
                self.spikes = torch.from_numpy(spikes_out).unsqueeze(1)
                # Les états V/g/refr sont écrits en place par le kernel.
            else:
                # 2) α-synapse : incrément w au spike présynaptique, décroissance τ_syn
                inc = torch.sparse.mm(self.W, delayed)
                self.g.mul_(self._syn_decay).add_(inc)
                # 3) intégration LIF + réfractaire absolu (in-place, zéro alloc
                # intermédiaire : I_syn est plié dans la mise à jour de V)
                self.V += (-(self.V - self.V_rest) + self.neuromod * self.g) * (self.dt / self.tau_m)
                sp = (self.V >= self.V_th) & (self._refr <= 0)
                self.V.masked_fill_(sp, self.V_reset)
                self._refr.sub_(1).clamp_(min=0)
                self._refr.masked_fill_(sp, self.refractory_steps)
                self.spikes = sp.float()
            # Stimuli externes convertis en Poisson (codage en fréquence).
            # I_full est nul hors sensoriels : bruit restreint aux ~16k
            # sensoriels (11× moins de tirages que le plein format n×B).
            if I_full is not None:
                Is = I_full[self._sensory_idx]
                d = (torch.rand(Is.shape, device=Is.device) <
                     (Is.clamp(min=0) * self.stim_p_per_mv)).float()
                self.spikes[self._sensory_idx] = torch.clamp(
                    self.spikes[self._sensory_idx] + d, max=1.0)
            # Réflexe d'alimentation : drive Poisson des GRN sucrées
            if self._sugar_drive_left > 0:
                drive = (torch.rand(self.n, B, device=self.device) <
                         self._sugar_hz * self.dt / 1000.0).float()
                self.spikes[self._sugar_idx, :] = torch.maximum(
                    self.spikes[self._sugar_idx, :], drive[self._sugar_idx, :])
                self._sugar_drive_left -= 1
            # Lésion : zéro spike, et le buffer de délai (qui vient d'être
            # alimenté avec les spikes courants) est nettoyé aussi.
            if self._lesion_idx is not None:
                self.spikes[self._lesion_idx, :] = 0
                self._delay[-1][self._lesion_idx, :] = 0
            self.t += 1
            return self.spikes

        if self._native is not None:
            V_np = self.V.numpy()
            sp_np = self.spikes.numpy()
            isyn = self._isyn_bufs.get(B)
            if isyn is None:
                isyn = np.zeros((self.n, B), dtype=np.float32)
                self._isyn_bufs[B] = isyn
            self._native.step(
                self._indptr, self._indices, self._data,
                sp_np, V_np, isyn, I_full,
                self.neuromod, self.tau_m, self.dt, self.V_th, self.V_reset,
                sp_np,  # spikes_out aliasé : la passe 1 lit tout avant la passe 2
            )
            if self._lesion_idx is not None:
                # sp_np est aliasé dans self.spikes : le masquage torch agit
                # en place et le kernel lira des spikes nuls au prochain pas.
                self.spikes[self._lesion_idx, :] = 0
            self.t += 1
            return self.spikes

        # --- Backend torch sparse CSR ---
        I_syn = self.neuromod * torch.sparse.mm(self.W, self.spikes)
        if I_full is not None:
            I_syn = I_syn + I_full

        # Intégration LIF (Euler)
        dV = (-self.V + I_syn) / self.tau_m * self.dt
        self.V = self.V + dV

        # Détection des spikes + réinitialisation
        spiked = self.V >= self.V_th
        self.V = torch.where(spiked, torch.full_like(self.V, self.V_reset), self.V)
        self.spikes = spiked.float()
        if self._lesion_idx is not None:
            self.spikes[self._lesion_idx, :] = 0
        self.t += 1

        return self.spikes

    @torch.no_grad()
    def step(self, I_ext: torch.Tensor | None = None) -> torch.Tensor:
        """
        Un pas de temps (1 ms) pour tout le batch.

        I_ext : (n_sensory,)   → même stimulus pour tout le batch
                (n_sensory, B) → un stimulus par élément du batch
                (n, B)         → courants déjà placés neurone par neurone
        Retourne : (n, B) spikes de ce pas (float 0/1).

        Deux backends : kernel C natif (CPU, vérifié bit-exact sur 1M réseaux)
        sinon torch sparse CSR.
        """
        if self._native is not None:
            I_full = self._prepare_I_native(I_ext)
        else:
            I_full = self._prepare_I_torch(I_ext)
        return self._step_prepared(I_full)

    @torch.no_grad()
    def run(
        self,
        I_ext: torch.Tensor,
        n_steps: int = 500,
        record_every: int = 5,
        frame_callback=None,
        return_history: bool = False,
        collect_frames: bool = False,
    ) -> dict:
        """
        Simule n_steps pas pour tout le batch. Zéro transfert CPU↔GPU pendant
        la boucle (sauf frames si un callback est fourni, pour le streaming).

        I_ext : voir step(). B est déduit de la forme.
        frame_callback : callable(list[int]) appelé à chaque frame enregistrée
            (spikes de la colonne 0 du batch, pour le streaming WebSocket).
        return_history : si True, empile aussi la trajectoire lue à chaque
            pas enregistré → motor_hist (T, B, n_motor) + descending_hist
            (T, B, n_desc) sur device. Coût ~0 (indexation déjà faite pour
            motor_sum). Base du LoopedReadout (RAPPORT_INTEGRATION B2) :
            l'info peut être dans la dynamique, pas la moyenne.
        collect_frames : si True, collecte les indices de spikes de TOUTES
            les colonnes à chaque pas enregistré → frames_all [T][B]. Sert
            à rejouer les frames du coup gagnant SANS re-simuler (le replay
            coûtait une 2e simu complète pour rien).

        Retourne :
          - motor_mean : (B, n_motor) activité motrice moyenne (sur device)
          - descending_last : (B, n_desc) spikes descendants au dernier pas
          - (+ motor_hist / descending_hist si return_history)
          - (+ frames_all si collect_frames)
        """
        if I_ext.dim() == 1:
            I_ext = I_ext.unsqueeze(1)
        batch_size = I_ext.shape[1]
        self.reset(batch_size)

        # Le courant est CONSTANT pendant la simulation : préparation unique
        if self._native is not None:
            I_full = self._prepare_I_native(I_ext)
        else:
            I_full = self._prepare_I_torch(I_ext)

        motor_sum = torch.zeros(batch_size, int(self.is_motor.sum()), device=self.device)
        n_recorded = 0
        hist_m, hist_d = [] if return_history else None, [] if return_history else None
        frames_all = [] if collect_frames else None

        for step in range(n_steps):
            spiked = self._step_prepared(I_full)

            if step % record_every == 0:
                motor_sum += self.spikes[self.is_motor].T
                n_recorded += 1
                if return_history:
                    hist_m.append(self.spikes[self.is_motor].T.clone())
                    hist_d.append(self.spikes[self.is_descending].T.clone())
                if collect_frames:
                    cols = []
                    for bcol in range(batch_size):
                        # torch.where (pas nonzero+squeeze : INTERNAL ASSERT
                        # multithread connu de nonzero dans cette version torch)
                        idx = torch.where(spiked[:, bcol])[0]
                        if idx.numel() > 2000:
                            idx = idx[:2000]
                        cols.append(idx.cpu().tolist())
                    frames_all.append(cols)
                if frame_callback is not None:
                    idx = torch.where(spiked[:, 0])[0]
                    if idx.numel() > 2000:
                        idx = idx[:2000]
                    frame_callback(idx.cpu().tolist())

        motor_mean = motor_sum / max(n_recorded, 1)
        descending_last = self.spikes[self.is_descending].T

        out = {
            "motor_mean": motor_mean,
            "descending_last": descending_last,
        }
        if return_history:
            # (T, B, n) — même objet pour train et inférence (principe RLT B1)
            out["motor_hist"] = torch.stack(hist_m, dim=0) if hist_m else motor_mean.unsqueeze(0)
            out["descending_hist"] = torch.stack(hist_d, dim=0) if hist_d else descending_last.unsqueeze(0)
        if collect_frames:
            out["frames_all"] = frames_all  # [T][B] listes d'indices
        return out

    # --- Compatibilité : API historique mono-batch ---
    @torch.no_grad()
    def simulate(self, I_ext: torch.Tensor, n_steps: int = 500, record_every: int = 5):
        """
        Simule n_steps pas avec un seul batch.
        Retourne (motor_history np (T, n_motor), frames list[list[int]]).
        """
        self.reset(1)
        if I_ext.dim() == 1:
            I_ext = I_ext.unsqueeze(1)
        I_full = (self._prepare_I_native(I_ext) if self._native is not None
                  else self._prepare_I_torch(I_ext))
        motor_history = []
        frames = []
        for step in range(n_steps):
            spiked = self._step_prepared(I_full)
            if step % record_every == 0:
                motor_history.append(
                    self.spikes[self.is_motor, 0].detach().cpu().numpy()
                )
                idx = torch.where(spiked[:, 0])[0]
                if idx.numel() > 2000:
                    idx = idx[:2000]
                frames.append(idx.cpu().numpy().tolist())
        return np.array(motor_history), frames

    def get_motor_readout(self) -> torch.Tensor:
        """(B, n_motor) spikes moteurs du dernier pas."""
        return self.spikes[self.is_motor].T

    def get_descending_readout(self) -> torch.Tensor:
        """(B, n_desc) spikes descendants du dernier pas."""
        return self.spikes[self.is_descending].T


def build_prod_brain(conn, synapse_model: str | None = None, wsyn: float | None = None) -> "FlyBrain":
    """Cerveau du régime prod (même construction que main.py) : alpha/Shiu par
    défaut (FLY_SYNAPSE_MODEL, FLY_WSYN=0.164 calibré), GRN sucrées branchées.
    Train et inférence partagent le même régime (principe RLT B1) — le mode
    current + norme 450 reste silencieux sur MaleCNS (0 spike moteur à 200 pas)."""
    import os
    synapse_model = synapse_model or os.environ.get("FLY_SYNAPSE_MODEL", "alpha")
    kwargs: dict = dict(synapse_model=synapse_model)
    if synapse_model == "alpha":
        wsyn = wsyn if wsyn is not None else float(os.environ.get("FLY_WSYN", "0.164"))
        kwargs["weight_scale"] = conn["mean_abs_weight"] * 20.0 * wsyn
        kwargs.update(V_rest=-52.0, V_th=-45.0, V_reset=-52.0,
                      tau_syn=5.0, delay_steps=2, refractory_steps=3)
    brain = FlyBrain(conn["W"], conn["is_sensory"], conn["is_motor"],
                     conn["is_descending"], **kwargs)
    if conn.get("sugar_grn_idx") is not None and len(conn["sugar_grn_idx"]):
        brain.set_sugar_neurons(conn["sugar_grn_idx"])
    return brain
