// Tamagotchi panel: stats, care actions, fly mood.
// Actions are optimistic: the UI reacts instantly, then reconciles with the
// backend response (which may run a full brain simulation in the background).
const PetUI = {
  state: null,
  busy: false,      // one in-flight action at a time (server-side lock)
  lastActionMessage: null,

  async init() {
    document.getElementById('btn-feed').onclick = () => this.action('feed');
    document.getElementById('btn-pet-cuddle').onclick = () => this.action('pet');
    document.getElementById('btn-clean').onclick = () => this.action('clean');
    document.getElementById('btn-sleep').onclick = () => this.action('sleep-toggle');
    document.getElementById('btn-courtship').onclick = () => this.action('courtship');
    document.getElementById('btn-threat').onclick = () => this.action('threat');
    document.getElementById('btn-study').onclick = () => this.action('study');
    await this.refresh();
    // Passive decay: refresh every 30 s
    setInterval(() => this.refresh(), 30000);
  },

  async refresh() {
    if (this.busy) return;
    try {
      this.state = await api('/api/pet');
      this.render();
    } catch (e) { console.warn('pet unavailable', e); }
  },

  async action(action) {
    if (this.busy) return;
    this.busy = true;
    // 'sleep-toggle' maps to wake/sleep depending on current state
    const current = this.state && this.state.sleeping ? 'wake' : 'sleep';
    const body = { action: action === 'sleep-toggle' ? current : action };
    const btn = document.getElementById('btn-' +
      (action === 'sleep-toggle' ? 'sleep' : action));
    const msgEl = document.getElementById('pet-message');
    const original = btn.textContent;
    if (btn) { btn.disabled = true; btn.textContent = '…'; }
    try {
      const d = await api('/api/pet/action', body);
      this.state = d.pet;
      this.render();
      msgEl.textContent = d.message || this.state.mood_message;
      if (d.frames && d.frames.length && typeof connectome !== 'undefined') {
        playFrames(d.frames); // reaction plays in the connectome view
      }
    } catch (e) {
      msgEl.textContent = `Oops: ${e.message}`;
    } finally {
      this.busy = false;
      if (btn) { btn.disabled = false; btn.textContent = original; }
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
    sleepBtn.textContent = s.sleeping ? '☀️ Wake' : '💤 Sleep';

    document.getElementById('stat-mood').textContent = s.mood;

    const age = s.age_hours < 24
      ? `${s.age_hours.toFixed(0)} h`
      : `${(s.age_hours / 24).toFixed(1)} d`;
    document.getElementById('pet-meta').textContent =
      `Level ${s.level} · ${s.xp % 100}/100 XP · age ${age}` +
      (s.sleeping ? ' · recovering' : '') +
      (s.last_learned ? ` · 📚 ${s.last_learned.title.slice(0, 40)}` : '');
  },
};

window.PetUI = PetUI;