// ⚗️ Labo des lésions virtuelles : sélection d'une population, grisation
// dans le cerveau WebGL, test du réflexe d'alimentation à la demande.
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
              title="${g.n.toLocaleString('fr-FR')} neurones lésés — déficit prédit ${pred}">
        ${g.name}
        <small>${g.n.toLocaleString('fr-FR')} neur · déficit prédit ${pred}</small>
      </button>`;
    }).join('');
    el.querySelectorAll('.lab-group').forEach(btn => {
      btn.onclick = () => this.lesion(btn.dataset.key);
    });
    if (connectome) connectome.setLesion(window.__labDisplay);
    if (!this.current) {
      document.getElementById('lab-result').textContent = 'Cerveau intact.';
    }
  },

  async lesion(key) {
    try {
      const r = await api('/api/lab/lesion', { key });
      document.getElementById('lab-result').textContent =
        `⚗️ ${r.name} lésée (${r.lesioned.toLocaleString('fr-FR')} neurones) — ` +
        `la mouche joue et réagit avec ce handicap.`;
    } catch (e) {
      document.getElementById('lab-result').textContent = 'Erreur : ' + e.message;
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
    res.textContent = '🧪 Simulation du réflexe en cours… (drive 100 Hz sur ' +
      'les GRN sucrées, ~45 s sur le vrai connectome)';
    try {
      const r = await api('/api/lab/test-reflex', {});
      const pct = r.deficit !== null && r.deficit !== undefined
        ? (r.deficit * 100).toFixed(0) + ' %' : '—';
      res.innerHTML =
        `🧪 Réflexe MN9 : <b>${r.mn9_hz} Hz</b> (intact : ${r.baseline_hz} Hz) ` +
        `→ déficit <b>${pct}</b> · lésion : ${r.current || 'aucune'}`;
    } catch (e) {
      res.textContent = 'Erreur : ' + e.message;
    } finally {
      btn.disabled = false;
    }
  },
};

window.LabUI = LabUI;
