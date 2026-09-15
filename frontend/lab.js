// ⚗️ Virtual-lesion lab: pick a population, dim it in the WebGL brain,
// and test the feeding reflex on demand.
const LabUI = {
  current: null,
  _timer: null,

  async init() {
    document.getElementById('btn-lab-clear').onclick = () => this.clear();
    document.getElementById('btn-lab-test').onclick = () => this.testReflex();
    await this.refresh();
  },

  labName(g) {
    return T(`__lab:${g.key}`) !== `__lab:${g.key}` ? T(`__lab:${g.key}`) : g.name;
  },

  async refresh() {
    const el = document.getElementById('lab-groups');
    let info;
    try {
      info = await api('/api/lab', null, 20000);
    } catch (e) {
      el.innerHTML = '<span class="empty-note">Lab needs the backend — connect to lesion the brain.</span>';
      return;
    }
    this.current = info.current;
    window.__labCurrent = info.current
      ? this.labName(info.groups.find(g => g.key === info.current) || { key: info.current, name: info.current })
      : null;
    window.__labDisplay = info.display_indices || null;
    el.innerHTML = info.groups.map(g => {
      const pred = g.predicted_deficit !== null && g.predicted_deficit !== undefined
        ? (g.predicted_deficit * 100).toFixed(0) + ' %' : '—';
      return `
      <button class="lab-group ${info.current === g.key ? 'lesion-on' : ''}"
              data-key="${g.key}"
              title="${g.n.toLocaleString('en-US')} lesioned neurons — predicted deficit ${pred}">
        ${this.labName(g)}
        <small>${g.n.toLocaleString('en-US')} neur · pred. deficit ${pred}</small>
      </button>`;
    }).join('');
    el.querySelectorAll('.lab-group').forEach(btn => {
      btn.onclick = () => this.lesion(btn.dataset.key);
    });
    if (connectome) connectome.setLesion(window.__labDisplay);
    if (!this.current) {
      document.getElementById('lab-result').textContent = 'Brain intact.';
    }
  },

  async lesion(key) {
    try {
      const r = await api('/api/lab/lesion', { key });
      document.getElementById('lab-result').textContent =
        `⚗️ ${this.labName({ key, name: r.name })} lesioned (${r.lesioned.toLocaleString('en-US')} neurons) — ` +
        `the fly plays and reacts with this handicap.`;
      toast(`⚗️ Lesion applied: ${this.labName({ key, name: r.name })}`, 'info');
    } catch (e) {
      document.getElementById('lab-result').textContent = 'Error: ' + e.message;
      toast(`Lesion failed: ${e.message}`, 'err');
    }
    await this.refresh();
  },

  async clear() {
    try {
      await api('/api/lab/clear', {});
      toast('🩹 Brain healed', 'ok', 2200);
    } catch (e) {
      toast(`Heal failed: ${e.message}`, 'err');
    }
    await this.refresh();
  },

  async testReflex() {
    const btn = document.getElementById('btn-lab-test');
    const res = document.getElementById('lab-result');
    setBusy(btn, true);
    const t0 = Date.now();
    res.textContent = '🧪 Running reflex simulation… (100 Hz drive on the sweet GRNs)';
    clearInterval(this._timer);
    this._timer = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      res.textContent = `🧪 Running reflex simulation… ${s}s elapsed (full connectome, please wait)`;
    }, 1000);
    try {
      const r = await api('/api/lab/test-reflex', {}, 600000);
      const pct = r.deficit !== null && r.deficit !== undefined
        ? (r.deficit * 100).toFixed(0) + ' %' : '—';
      res.innerHTML =
        `🧪 MN9 reflex: <b>${r.mn9_hz} Hz</b> (intact: ${r.baseline_hz} Hz) ` +
        `→ deficit <b>${pct}</b> · lesion: ${r.current || 'none'}`;
      toast(`🧪 Reflex: ${r.mn9_hz} Hz, deficit ${pct}`, 'ok');
    } catch (e) {
      res.textContent = 'Error: ' + e.message;
      toast(`Reflex test failed: ${e.message}`, 'err');
    } finally {
      clearInterval(this._timer);
      setBusy(btn, false);
    }
  },
};

window.LabUI = LabUI;
