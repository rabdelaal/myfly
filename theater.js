// Théâtre du cerveau : plein écran, simulation continue en WebSocket,
// orbite automatique, synapses vivantes, HUD de fréquences et légendes.
const Theater = {
  ws: null,
  view: null,

  open() {
    const overlay = document.getElementById('theater');
    overlay.classList.remove('hidden');
    const canvas = document.getElementById('theater-canvas');
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;

    const data = window.CONNECTOME_DATA || {};
    const n = data.positions ? data.positions.length : 5000;
    try {
      this.view = new ConnectomeView(canvas, n,
                                     data.positions || null,
                                     data.category || null,
                                     data.hotspot || null);
    } catch (e) {
      document.getElementById('theater-caption').textContent =
        'WebGL unavailable: ' + e.message;
      return;
    }
    if (data.edges && data.edges.a) this.view.setEdges(data.edges);
    this.view.autoOrbit = true;
    // ⚗️ Labo : appliquer la lésion en cours (neurones grisés)
    if (window.__labDisplay && this.view.setLesion) {
      this.view.setLesion(window.__labDisplay);
    }

    document.getElementById('theater-close').onclick = () => this.close();
    window.addEventListener('resize', this._onResize = () => {
      canvas.width = window.innerWidth * window.devicePixelRatio;
      canvas.height = window.innerHeight * window.devicePixelRatio;
      this.view.resize();
    });

    if (window.LOCAL_MODE) {
      document.getElementById('theater-caption').textContent =
        'Demo mode — orbiting only, no live spikes without backend';
      const cap = document.getElementById('theater-caption');
      cap.classList.remove('caption-show');
      void cap.offsetWidth;
      cap.classList.add('caption-show');
      return; // pas de WebSocket en local : orbite seule, zéro erreur console
    }
    const proto = (window.FLY_API || '').replace(/^http/, 'ws');
    this.ws = new WebSocket(proto + '/ws/live');
    this.ws.onmessage = (ev) => this._onFrame(JSON.parse(ev.data));
    this.ws.onclose = () => { if (!overlay.classList.contains('hidden')) {
      document.getElementById('theater-caption').textContent = 'Connection lost…';
    }};
  },

  _onFrame(m) {
    if (m.spikes) this.view.setActivation(m.spikes);
    document.getElementById('th-hz-sensory').textContent = m.hz.sensory.toFixed(1);
    document.getElementById('th-hz-inter').textContent = m.hz.inter.toFixed(1);
    document.getElementById('th-hz-motor').textContent = m.hz.motor.toFixed(1);
    document.getElementById('th-hz-desc').textContent = m.hz.descending.toFixed(1);
    // Populations dimorphes (absentes si vieux backend → on cache la ligne)
    for (const [id, key] of [['th-hz-hotspot', 'hotspot'], ['th-hz-fru', 'fru']]) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (m.hz[key] === undefined) { el.closest('div').style.display = 'none'; continue; }
      el.textContent = m.hz[key].toFixed(1);
    }
    document.getElementById('th-octo').textContent = '×' + m.neuromod.toFixed(2);
    const moodEl = document.getElementById('th-mood');
    moodEl.textContent = m.sleeping ? '💤 sleeping' : T(m.mood);

    if (m.caption) {
      const cap = document.getElementById('theater-caption');
      cap.textContent = m.caption;
      cap.classList.remove('caption-show');
      void cap.offsetWidth; // relance l'animation CSS
      cap.classList.add('caption-show');
    }
  },

  close() {
    if (this.ws) { this.ws.close(); this.ws = null; }
    window.removeEventListener('resize', this._onResize);
    document.getElementById('theater').classList.add('hidden');
    if (this.view && this.view.destroy) this.view.destroy(); // stoppe la boucle rAF
    this.view = null;
  },
};

window.Theater = Theater;
