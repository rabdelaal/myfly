// 📚 Mind tab: teach the fly (learn), search its memory, browse knowledge.
const MindUI = {
  async init() {
    document.getElementById('btn-learn').onclick = () => this.learn();
    document.getElementById('btn-search').onclick = () => this.search();
    document.getElementById('learn-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.learn();
    });
    document.getElementById('search-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.search();
    });
    await this.refresh();
  },

  async refresh() {
    const el = document.getElementById('knowledge-list');
    let items = [];
    try {
      const r = await api('/api/knowledge', null, 20000);
      items = r.items || [];
    } catch (e) {
      el.innerHTML = '<span class="empty-note">Knowledge needs the backend.</span>';
      return;
    }
    if (!items.length) {
      el.innerHTML = '<span class="empty-note">Nothing learned yet — teach it above.</span>';
      return;
    }
    el.innerHTML = '<ul>' + items.slice(0, 12).map(it =>
      `<li><a href="${it.source}" target="_blank" rel="noopener">${escapeHtml(it.title)}</a>` +
      `<div class="src">${escapeHtml(it.source)} · ${new Date(it.mtime * 1000).toLocaleDateString('en-US')}</div></li>`
    ).join('') + '</ul>';
  },

  async learn() {
    const input = document.getElementById('learn-input');
    const btn = document.getElementById('btn-learn');
    const q = input.value.trim();
    if (!q) { toast('Type a topic first', 'err'); return; }
    setBusy(btn, true);
    try {
      const r = await api('/api/learn', { query: q, num: 2 }, 300000);
      toast(`📚 Learned ${r.learned} page${r.learned > 1 ? 's' : ''} about “${q}”`, 'ok');
      input.value = '';
      await this.refresh();
      if (window.PetUI) await PetUI.refresh();
    } catch (e) {
      toast(`Learn failed: ${e.message}`, 'err');
    } finally {
      setBusy(btn, false);
    }
  },

  async search() {
    const input = document.getElementById('search-input');
    const btn = document.getElementById('btn-search');
    const box = document.getElementById('search-results');
    const q = input.value.trim();
    if (!q) { toast('Type something to recall', 'err'); return; }
    setBusy(btn, true);
    box.innerHTML = '<div class="skeleton"></div>';
    try {
      const r = await api(`/api/search?q=${encodeURIComponent(q)}&top=3`, null, 60000);
      if (!r.items || !r.items.length) {
        box.innerHTML = '<span class="empty-note">No memory of that yet — teach it first.</span>';
      } else {
        box.innerHTML = r.items.map(it =>
          `<div class="mind-card"><h4>${escapeHtml(it.title)}</h4>` +
          `<p>${escapeHtml((it.excerpt || '').slice(0, 220))}…</p>` +
          `<div class="src">overlap ${it.overlap} · <a href="${it.source}" target="_blank" rel="noopener">source</a></div></div>`
        ).join('');
      }
    } catch (e) {
      box.innerHTML = `<span class="empty-note">Recall failed: ${escapeHtml(e.message)}</span>`;
    } finally {
      setBusy(btn, false);
    }
  },
};

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

window.MindUI = MindUI;
