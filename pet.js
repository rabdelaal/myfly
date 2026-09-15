// Tamagotchi panel: stats, care actions, fly mood.
// Actions are optimistic: the UI reacts instantly, then reconciles with the
// backend response (which may run a full brain simulation in the background).
const MOOD_HUE = {
  sleeping: 220, delighted: 150, ravie: 150, content: 185, contente: 185,
  grumpy: 35, grognon: 35, starving: 0, 'affamée': 0, miserable: 340,
  misérable: 340, exhausted: 265, 'épuisée': 265, itchy: 300, inconfortable: 300,
};
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
      this.state = await api('/api/pet', null, 15000);
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
    if (btn) { btn.disabled = true; btn.classList.add('busy'); }
    try {
      const d = await api('/api/pet/action', body);
      this.state = d.pet;
      this.render();
      const msg = T(d.message) || T(this.state.mood_message);
      // Citation study : titre + extrait sur deux lignes
      if (msg.includes('📚')) {
        const [head, ...rest] = msg.split('\n');
        msgEl.innerHTML = '';
        msgEl.append(document.createTextNode(head));
        if (rest.length) {
          const cite = document.createElement('span');
          cite.className = 'cite';
          cite.textContent = rest.join(' ');
          msgEl.append(cite);
        }
      } else {
        msgEl.textContent = msg;
      }
      if (d.frames && d.frames.length && typeof connectome !== 'undefined') {
        playFrames(d.frames); // reaction plays in the connectome view
      }
    } catch (e) {
      msgEl.textContent = `Oops: ${e.message}`;
      toast(`Care action failed: ${e.message}`, 'err');
    } finally {
      this.busy = false;
      if (btn) { btn.disabled = false; btn.classList.remove('busy'); }
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
        const cls = v >= 50 ? 'good' : v >= 25 ? 'warn' : 'bad';
        if (!bar.classList.contains(cls)) {
          bar.className = cls;
          bar.classList.add('tick');
          setTimeout(() => bar.classList.remove('tick'), 500);
        }
      }
      if (val) val.textContent = Math.round(v);
    }

    const msg = document.getElementById('pet-message');
    if (this.lastActionMessage) {
      msg.textContent = T(this.lastActionMessage);
      this.lastActionMessage = null;
    } else if (!msg.querySelector('.cite')) {
      msg.textContent = T(s.mood_message);
    }

    const emoji = document.getElementById('fly-emoji');
    const stage = document.getElementById('fly-stage');
    const zzz = document.getElementById('fly-zzz');
    stage.className = s.sleeping ? 'sleeping' : `mood-${s.mood}`;
    const hue = MOOD_HUE[s.sleeping ? 'sleeping' : s.mood] ?? 190;
    stage.style.setProperty('--mood-hue', hue);
    zzz.style.display = s.sleeping ? 'inline' : 'none';
    emoji.style.filter = s.stats.hygiene < 30 ? 'grayscale(0.7) brightness(0.8)' : '';

    const sleepBtn = document.getElementById('btn-sleep');
    sleepBtn.textContent = s.sleeping ? '☀️ Wake' : '💤 Sleep';

    const moodEn = T(s.mood);
    document.getElementById('stat-mood').textContent = moodEn;
    const hudMood = document.getElementById('hud-mood');
    if (hudMood) hudMood.textContent = (s.sleeping ? '💤 ' : '') + moodEn;
    const hudLesion = document.getElementById('hud-lesion');
    if (hudLesion && window.__labCurrent) {
      hudLesion.style.display = '';
      document.getElementById('hud-lesion-name').textContent = window.__labCurrent;
    } else if (hudLesion) {
      hudLesion.style.display = 'none';
    }

    const age = s.age_hours < 24
      ? `${s.age_hours.toFixed(0)} h`
      : `${(s.age_hours / 24).toFixed(1)} d`;
    // Anneau XP : progression vers le niveau suivant
    const arc = document.getElementById('xp-arc');
    if (arc) {
      const frac = (s.xp % 100) / 100;
      arc.style.strokeDashoffset = String(81.7 * (1 - frac));
    }
    document.getElementById('pet-meta').textContent =
      `Level ${s.level} · ${s.xp % 100}/100 XP · age ${age}` +
      (s.sleeping ? ' · recovering' : '') +
      (s.last_learned ? ` · 📚 ${s.last_learned.title.slice(0, 40)}` : '');
  },
};

window.PetUI = PetUI;
