// ⚗️ Virtual-lesion lab: pick a population, gray it out in the WebGL brain,
// and test the feeding reflex on demand.
const LabUI = {
  current: null,

  async init() {
    document.getElementById('btn-lab-clear').onclick = () => this.clear();
    document.getElementById('btn-lab-test').onclick = () => this.testReflex();
    await this.refresh();
  },

  async refresh() {
    const info = await api('/api/lab');
    this.current = info.current;
    window.__labDisplay = info.display_indices || null;
    const el = document.getElementById('lab-groups');
    el.innerHTML = info.groups.map(g => {
      const pred = g.predicted_deficit !== null && g.predicted_deficit !== undefined
        ? (g.predicted_deficit * 100).toFixed(0) + ' %' : '—';
      return `
      <button class="lab-group ${info.current === g.key ? 'lesion-on' : ''}"
              data-key="${g.key}"
              title="${g.n.toLocaleString('en-US')} lesioned neurons — predicted deficit ${pred}">
        ${g.name}
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
        `⚗️ ${r.name} lesioned (${r.lesioned.toLocaleString('en-US')} neurons) — ` +
        `the fly plays and reacts with this handicap.`;
    } catch (e) {
      document.getElementById('lab-result').textContent = 'Error: ' + e.message;
    }
    await this.refresh();
  },

  async clear() {
    await api('/api/lab/clear', {});
    await this.refresh();
  },

  async testReflex() {
    const btn = document.getElementById('btn-lab-test');
    const res = document.getElementById('lab-result');
    btn.disabled = true;
    res.textContent = '🧪 Running reflex simulation… (100 Hz drive on the ' +
      'sweet GRNs, ~45 s on the full connectome)';
    try {
      const r = await api('/api/lab/test-reflex', {});
      const pct = r.deficit !== null && r.deficit !== undefined
        ? (r.deficit * 100).toFixed(0) + ' %' : '—';
      res.innerHTML =
        `🧪 MN9 reflex: <b>${r.mn9_hz} Hz</b> (intact: ${r.baseline_hz} Hz) ` +
        `→ deficit <b>${pct}</b> · lesion: ${r.current || 'none'}`;
    } catch (e) {
      res.textContent = 'Error: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  },
};

window.LabUI = LabUI;
