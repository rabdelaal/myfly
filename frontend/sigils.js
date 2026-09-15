// Sigils showcase: pick a topology, hear its rhythm (WebAudio), see metrics.
// Mapping mirrors backend/sigil_music.py: domHz -> pitch, burst -> velocity,
// MC -> phrase length. No MIDI file needed: oscillators directly.
const Sigils = {
  data: {},
  playing: false,
  ctx: null,

  async init() {
    try {
      this.data = await api('/api/sigils');
    } catch (e) { console.warn('sigils unavailable', e); return; }
    const sel = document.getElementById('sigil-select');
    sel.innerHTML = '';
    for (const [name, m] of Object.entries(this.data)) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = `${name} (MC ${m.MC})`;
      sel.appendChild(o);
    }
    sel.value = 'flower';
    sel.onchange = () => this.show();
    document.getElementById('btn-sigil-play').onclick = () => this.play();
    document.getElementById('btn-sigil-stop').onclick = () => this.stop();
    this.show();
  },

  show() {
    const name = document.getElementById('sigil-select').value;
    const m = this.data[name];
    if (!m) return;
    document.getElementById('sigil-metrics').textContent =
      `${name}: ${m.edges} edges · ${m.kpas} kpas/s · rate ${m.rate} · ` +
      `burst ${m.burst} · dom ${m.domHz} Hz · MC ${m.MC}`;
    document.getElementById('sigil-gait').textContent = '';
  },

  phrase(name) {
    const m = this.data[name];
    const base = 48 + Math.floor(m.domHz * 40) % 24;
    const vel = Math.max(40, Math.min(120, 60 + Math.floor(m.burst)));
    const len = 8 + Math.floor(m.MC * 20);
    const step = (m.id % 5) - 2;
    const evts = [];
    let p = base;
    for (let i = 0; i < len; i++) {
      p = Math.max(36, Math.min(84, p + step + ((i * 7) % 3 - 1)));
      evts.push({ pitch: p, vel, dur: 0.22 });
    }
    return evts;
  },

  midi2freq(pitch) {
    return 440 * Math.pow(2, (pitch - 69) / 12);
  },

  async play() {
    this.stop();
    const name = document.getElementById('sigil-select').value;
    const evts = this.phrase(name);
    this.ctx = this.ctx ?? new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();
    this.playing = true;
    let t = this.ctx.currentTime + 0.05;
    const gait = [];
    for (const [i, e] of evts.entries()) {
      if (!this.playing) break;
      const o = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      o.type = i % 4 === 0 ? 'triangle' : 'sine';
      o.frequency.value = this.midi2freq(e.pitch);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(e.vel / 127 * 0.4, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + e.dur);
      o.connect(g).connect(this.ctx.destination);
      o.start(t);
      o.stop(t + e.dur + 0.05);
      gait.push(`${e.pitch}`);
      t += e.dur;
    }
    document.getElementById('sigil-gait').textContent = '♪ ' + gait.join(' · ');
    setTimeout(() => { this.playing = false; }, (t - this.ctx.currentTime) * 1000);
  },

  stop() {
    this.playing = false;
    if (this.ctx) this.ctx.close().catch(() => {});
    this.ctx = null;
  },
};
