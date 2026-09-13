// Panneau Tamagotchi : stats, actions de soin, humeur de la mouche.
const PetUI = {
  state: null,
  pending: false,

  async init() {
    document.getElementById('btn-feed').onclick = () => this.action('feed');
    document.getElementById('btn-pet-cuddle').onclick = () => this.action('pet');
    document.getElementById('btn-clean').onclick = () => this.action('clean');
    document.getElementById('btn-sleep').onclick = () => this.action('sleep-toggle');
    document.getElementById('btn-courtship').onclick = () => this.action('courtship');
    document.getElementById('btn-threat').onclick = () => this.action('threat');
    await this.refresh();
    // Décroissance passive : rafraîchir toutes les 30 s
    setInterval(() => this.refresh(), 30000);
  },

  async refresh() {
    if (this.pending) return;
    try {
      this.state = await api('/api/pet');
      this.render();
    } catch (e) { console.warn('pet indisponible', e); }
  },

  async action(action) {
    if (this.pending) return;
    this.pending = true;
    const msgEl = document.getElementById('pet-message');
    try {
      // "sleep-toggle" : le backend attend feed/pet/clean/sleep/wake
      const current = this.state && this.state.sleeping ? 'wake' : 'sleep';
      const body = { action: action === 'sleep-toggle' ? current : action };
      const d = await api('/api/pet/action', body);
      this.state = d.pet;
      this.render();
      msgEl.textContent = d.message || this.state.mood_message;
      if (d.frames && d.frames.length && typeof connectome !== 'undefined') {
        playFrames(d.frames); // la réaction s'affiche dans le connectome
      }
    } catch (e) {
      msgEl.textContent = `Oups : ${e.message}`;
    } finally {
      this.pending = false;
    }
  },

  render() {
    const s = this.state;
    if (!s) return;
    for (const key of Object.keys(s.stats)) {
      const v = s.stats[key];
      const bar = document.getElementById(`bar-${key}`);
      const val = document.getElementById(`val-${key}`);
      if (bar) {
        bar.style.width = `${v}%`;
        bar.className = v >= 50 ? 'good' : v >= 25 ? 'warn' : 'bad';
      }
      if (val) val.textContent = Math.round(v);
    }

    const msg = document.getElementById('pet-message');
    if (this.lastActionMessage) {
      msg.textContent = this.lastActionMessage;
      this.lastActionMessage = null;
    } else {
      msg.textContent = s.mood_message;
    }

    const emoji = document.getElementById('fly-emoji');
    const stage = document.getElementById('fly-stage');
    const zzz = document.getElementById('fly-zzz');
    stage.className = s.sleeping ? 'sleeping' : `mood-${s.mood}`;
    zzz.style.display = s.sleeping ? 'inline' : 'none';
    emoji.style.filter = s.stats.hygiene < 30 ? 'grayscale(0.7) brightness(0.8)' : '';

    const sleepBtn = document.getElementById('btn-sleep');
    sleepBtn.textContent = s.sleeping ? '☀️ Réveiller' : '💤 Dormir';

    document.getElementById('stat-mood').textContent = s.mood;

    const age = s.age_hours < 24
      ? `${s.age_hours.toFixed(0)} h`
      : `${(s.age_hours / 24).toFixed(1)} j`;
    document.getElementById('pet-meta').textContent =
      `Niveau ${s.level} · ${s.xp % 100}/100 XP · âge ${age}` +
      (s.sleeping ? ' · elle récupère des forces' : '');
  },
};

window.PetUI = PetUI;
