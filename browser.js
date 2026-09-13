// Mode démo 100 % navigateur (GitHub Pages, sans backend).
// Stratégie : même protocole que le backend (scores batchés, assist captures
// d'abord, réaction = activité du gagnant), mais cerveau synthétique 5k en
// Web Worker + règles chess.js. Le readout est NON entraîné (comme un backend
// frais) et le Tamagotchi vit en localStorage — pas de GRN/MN9, pas de
// dimorphisme réel : la fiche ♂ reste masquée, c'est assumé dans le bandeau.
const BrowserMode = {
  worker: null, game: null, pet: null, seq: 0, pending: new Map(),

  call(msg) {
    return new Promise((resolve) => {
      const id = ++this.seq;
      this.pending.set(id, resolve);
      this.worker.postMessage({ id, ...msg });
    });
  },

  async start({ connCanvas }) {
    window.LOCAL_MODE = true;
    document.getElementById('backend-status').textContent =
      'Mode démo locale — cerveau synthétique 5 000 neurones dans ton navigateur (sans backend)';
    const liveBtn = document.getElementById('btn-live');
    if (liveBtn) { liveBtn.disabled = true; liveBtn.title = 'Cerveau en direct = backend uniquement'; }
    const roomBox = document.querySelector('.room-controls');
    if (roomBox) roomBox.style.display = 'none';
    const rs = document.getElementById('room-status');
    if (rs) rs.textContent = 'Multiplayer rooms = backend only';

    this.worker = new Worker('flyworker.js');
    this.worker.onmessage = (e) => {
      const r = this.pending.get(e.data.id);
      if (r) { this.pending.delete(e.data.id); r(e.data); }
    };
    const info = await this.call({ type: 'init', n: 5000 });
    document.getElementById('stat-neurons').textContent = info.n.toLocaleString('fr-FR');
    document.getElementById('stat-synapses').textContent = info.nnz.toLocaleString('fr-FR');

    // Cerveau synthétique : positions aléatoires, catégories réelles
    const n = info.n, ni = n - info.ns - info.nm, descN = Math.min(100, ni);
    const cats = new Array(n).fill(0);
    for (let i = 0; i < info.ns; i++) cats[i] = 1;
    for (let i = n - info.nm; i < n; i++) cats[i] = 2;
    for (let i = n - info.nm - descN; i < n - info.nm; i++) cats[i] = 3;
    const ea = [], eb = [], es = [];
    for (let k = 0; k < 800; k++) {
      const a = (Math.random() * n) | 0, b = (Math.random() * n) | 0;
      if (a === b) continue;
      ea.push(a); eb.push(b); es.push(Math.random() < 0.8 ? 1 : 0);
    }
    connectome = new ConnectomeView(connCanvas, n, null, cats);
    connectome.setEdges({ a: ea, b: eb, sign: es });
    window.CONNECTOME_DATA = { positions: null, category: cats, edges: { a: ea, b: eb, sign: es } };

    this.game = new Chess();
    this.petLoad();
    PetUI.state = this.petStatus();
    PetUI.render();
    PetUI.refresh = async () => { this.petDecay(); PetUI.state = this.petStatus(); PetUI.render(); };
    PetUI.action = async (action) => this.petAction(action);
    document.getElementById('stat-mood').textContent = this.pet.mood;
    this.syncBoard();
    setStatus('Mode local : à toi (blancs), la mouche joue les noirs');
  },

  // --- Échecs locaux ---
  uciList(verbose) { return verbose.map(m => m.from + m.to + (m.promotion || '')); },

  syncBoard(extra) {
    const verbose = this.game.moves({ verbose: true });
    currentState = {
      fen: this.game.fen(),
      turn: this.game.turn() === 'w' ? 'white' : 'black',
      legal_moves: this.uciList(verbose),
      game_over: this.game.game_over(),
      result: this.game.game_over() ? (this.game.in_draw() ? '1/2-1/2' : (this.game.turn() === 'w' ? '0-1' : '1-0')) : null,
      ...extra,
    };
    board3d.setPosition(currentState.fen);
    deselect();
  },

  boardFeatures(fen, game) {
    // Même layout que encoding.py : 64×12 + extras (via chess.js)
    const pieces = [];
    const placement = fen.split(' ')[0];
    const order = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
    let sq = 56; // a8 en index 0..63 (a1=0 comme python-chess SQUARES)
    const put = (ch) => {
      const col = ch === ch.toUpperCase() ? 1 : 0;
      const file = sq % 8, rank = Math.floor(sq / 8);
      const pysq = rank * 8 + file; // python-chess : a1=0..h8=63, rangs bottom-up
      pieces.push([pysq, order[ch.toLowerCase()], col]);
      sq++;
    };
    for (const ch of placement) {
      if (ch === '/') { sq -= 16; continue; }
      if (/\d/.test(ch)) { sq += parseInt(ch); continue; }
      put(ch);
    }
    const vals = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
    let material = 0;
    for (const [s, pt, col] of pieces) material += vals[['', 'p', 'n', 'b', 'r', 'q', 'k'][pt]] * (col ? 1 : -1);
    return {
      pieces, legal: game.moves().length, check: game.in_check() ? 1 : 0,
      mate: game.in_checkmate() ? 1 : 0, stale: game.in_stalemate() ? 1 : 0,
      turn: game.turn() === 'w' ? 1 : 0, fullmove: parseInt(fen.split(' ').pop()) || 1, material,
    };
  },

  async playMove(uci) {
    if (busy) return;
    busy = true;
    setStatus('The fly is thinking (local brain)…');
    try {
      const mv = this.game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' });
      if (!mv) { busy = false; return; }
      if (this.game.game_over()) { this.afterHuman(); return; }
      // Candidats : captures d'abord (règle assist backend), max 8
      const verbose = this.game.moves({ verbose: true });
      const caps = verbose.filter(m => m.flags.includes('c') || m.flags.includes('e'));
      const others = verbose.filter(m => !(m.flags.includes('c') || m.flags.includes('e')));
      const cands = [...caps, ...others].slice(0, 8);
      const boards = cands.map(m => {
        const g = new Chess(this.game.fen());
        g.move({ from: m.from, to: m.to, promotion: m.promotion || 'q' });
        return this.boardFeatures(g.fen(), g);
      });
      const r = await this.call({ type: 'score', boards, steps: 60, record: 10 });
      const scored = cands.map((m, i) => ({ uci: m.from + m.to + (m.promotion || ''), s: r.scores[i] }));
      const pool = caps.length ? scored.filter(s => caps.some(m => (m.from + m.to + (m.promotion || '')) === s.uci)) : scored;
      const best = pool.reduce((a, b) => (b.s > a.s ? b : a));
      const bm = cands.find(m => (m.from + m.to + (m.promotion || '')) === best.uci);
      this.game.move({ from: bm.from, to: bm.to, promotion: bm.promotion || 'q' });
      const col = cands.indexOf(bm);
      const frames = r.framesAll.map(fr => fr[col]);
      const scores = Object.fromEntries(scored.map(s => [s.uci, s.s]));
      const motor = r.motorMeans[col];
      this.afterHuman({ scores, frames, motor });
      if (window.Theater && Theater.view) {
        let i = 0;
        const t = setInterval(() => {
          if (i >= frames.length || !Theater.view) { clearInterval(t); return; }
          Theater.view.setActivation(frames[i++]);
        }, 60);
      }
    } finally {
      busy = false;
    }
  },

  afterHuman(fly) {
    this.petDecay();
    this.pet.stats.energy = Math.max(0, this.pet.stats.energy - 3);
    this.pet.stats.satiety = Math.max(0, this.pet.stats.satiety - 2);
    this.pet.stats.happiness = Math.min(100, this.pet.stats.happiness + 2);
    this.pet.xp += 3;
    this.petSave();
    let reaction = null;
    if (fly) {
      const strength = Math.tanh(fly.motor * 300);
      reaction = {
        strength: +strength.toFixed(3),
        message: strength >= 0.35
          ? 'Ses lobes optiques s\'affolent sur l\'échiquier : elle ADORE ça ♟️'
          : 'Elle regarde l\'échiquier sans grande conviction.',
      };
    }
    this.syncBoard(fly ? { scores: fly.scores, frames: fly.frames, pet_reaction: reaction } : null);
    if (fly && reaction) {
      document.getElementById('pet-message').textContent = reaction.message;
      if (fly.frames) playFrames(fly.frames);
    }
    if (fly && fly.scores) renderCandidates(fly.scores);
    PetUI.state = this.petStatus();
    PetUI.render();
    if (!currentState.game_over && currentState.turn === 'white') setStatus('À toi — clique une pièce');
    else if (!currentState.game_over) setStatus('Trait aux noirs…');
    else setStatus('Game over: ' + (currentState.result || '—'));
  },

  // --- Tamagotchi local (mêmes taux que backend, état en localStorage) ---
  petLoad() {
    try {
      const s = JSON.parse(localStorage.getItem('flylocal1'));
      if (s && s.stats) { this.pet = s; this.petDecay(); return; }
    } catch (e) { /* état frais */ }
    this.pet = { stats: { satiety: 80, happiness: 80, energy: 80, hygiene: 80 }, xp: 0, sleeping: false, last_tick: Date.now() / 1000, birth: Date.now() / 1000 };
  },
  petSave() {
    try { localStorage.setItem('flylocal1', JSON.stringify(this.pet)); } catch (e) { /* privé */ }
  },
  petDecay() {
    const now = Date.now() / 1000;
    const h = Math.min((now - this.pet.last_tick) / 3600, 48);
    if (h <= 0) { this.pet.last_tick = now; return; }
    const R = { satiety: 9, happiness: 6, energy: 7, hygiene: 5 };
    if (this.pet.sleeping) {
      this.pet.stats.energy = Math.min(100, this.pet.stats.energy + 30 * h);
      this.pet.stats.satiety = Math.max(0, this.pet.stats.satiety - 4.5 * h);
      this.pet.stats.happiness = Math.max(0, this.pet.stats.happiness - 3 * h);
      if (this.pet.stats.energy >= 99 || h >= 8) this.pet.sleeping = false;
    } else {
      for (const k of Object.keys(R)) this.pet.stats[k] = Math.max(0, this.pet.stats[k] - R[k] * h);
      if (this.pet.stats.energy <= 0.5) this.pet.sleeping = true;
    }
    this.pet.last_tick = now;
    this.petSave();
  },
  petMood() {
    if (this.pet.sleeping) return 'dort';
    const s = this.pet.stats, low = Object.keys(s).reduce((a, b) => (s[a] < s[b] ? a : b));
    if (s[low] < 20) return { satiety: 'affamée', happiness: 'misérable', energy: 'épuisée', hygiene: 'inconfortable' }[low];
    if (s.happiness >= 75) return 'ravie';
    if (s.happiness >= 50) return 'contente';
    return 'grognon';
  },
  petStatus() {
    this.pet.mood = this.petMood();
    const MM = {
      dort: 'Chut… elle dort profondément 💤', ravie: 'Elle bourdonne de bonheur autour de son terrarium 🪰✨',
      contente: 'Elle se lisse les antennes, l\'air tranquille.', grognon: 'Elle boude dans un coin du terrarium 😤',
      affamée: 'Elle tourne en rond : elle crève de faim ! 🍯', misérable: 'Elle a l\'air vraiment triste…',
      épuisée: 'Ses ailes traînent : elle est épuisée 💤', inconfortable: 'Elle se démange : une petite toilette s\'impose 🚿',
    };
    const wb = (this.pet.stats.satiety + this.pet.stats.happiness + this.pet.stats.energy + this.pet.stats.hygiene) / 400;
    return {
      stats: { ...this.pet.stats }, sleeping: this.pet.sleeping, mood: this.pet.mood,
      mood_message: MM[this.pet.mood], wellbeing: +wb.toFixed(3),
      chess_noise: +((1 - wb) * 0.4).toFixed(3), level: Math.floor(this.pet.xp / 100) + 1,
      xp: this.pet.xp, xp_next: (Math.floor(this.pet.xp / 100) + 1) * 100,
      age_hours: +((Date.now() / 1000 - this.pet.birth) / 3600).toFixed(1),
    };
  },
  async petAction(action) {
    const msgEl = document.getElementById('pet-message');
    this.petDecay();
    const act = action === 'sleep-toggle' ? (this.pet.sleeping ? 'wake' : 'sleep') : action;
    const EFF = {
      feed: { satiety: 30, hygiene: -6, happiness: 6 }, pet: { happiness: 8, energy: -1 },
      clean: { hygiene: 100, happiness: -4 }, courtship: { happiness: 10, energy: -8 },
      threat: { happiness: -6, energy: -5 },
    };
    const MSG = {
      feed: ['Gloup ! Elle se jette sur le sirop, ailes vibrantes de plaisir 🍯', 'Elle goûte distraitement quelques gouttes…'],
      pet: ['Elle frotte ses pattes avant, visiblement ravie d\'être caressée 🥰', 'Un petit frémissement antennaire, elle te tolère.'],
      clean: ['Pfuit ! Éclaboussée mais toute propre, elle s\'essuie avec énergie 🚿', 'Elle survit au bain avec dignité.'],
      wake: ['Elle s\'étire, déploie ses ailes et bourdonne : prête ! ☀️', 'Elle ouvre un œil… grognonne… mais se lève.'],
      courtship: ['Il déploie une aile et chante : parade nuptiale en cours 🎻🪰', 'Un petit frétillement d\'aile, timide…'],
      threat: ['Il fonce pattes en avant, ailes écartées : intimidation maximale 😠', 'Il fait un pas menaçant puis hésite.'],
    };
    if (act === 'sleep') {
      if (this.pet.sleeping) { msgEl.textContent = 'Elle dort déjà ! 💤'; return; }
      this.pet.sleeping = true; this.pet.xp += 2; this.petSave();
      msgEl.textContent = 'Les lumières s\'éteignent… bonne nuit 🌙';
    } else if (act === 'wake') {
      if (!this.pet.sleeping) { msgEl.textContent = 'Elle est déjà réveillée !'; return; }
      this.pet.sleeping = false; this.pet.xp += 2; this.petSave();
      msgEl.textContent = 'Elle s\'étire… prête ! ☀️';
    } else {
      if (this.pet.sleeping) { msgEl.textContent = 'Chut ! Elle dort. Réveille-la d\'abord 💤'; return; }
      for (const [k, d] of Object.entries(EFF[act] || {})) {
        if (act === 'clean' && k === 'hygiene') this.pet.stats[k] = 100;
        else this.pet.stats[k] = Math.max(0, Math.min(100, this.pet.stats[k] + d));
      }
      this.pet.xp += 5;
      this.petSave();
      setStatus('La mouche sent… (cerveau local)');
      const r = await this.call({ type: 'react', action: act, steps: 60 });
      const [strong, weak] = MSG[act] || MSG.pet;
      msgEl.textContent = r.strength >= 0.35 ? strong : weak;
    }
    PetUI.state = this.petStatus();
    PetUI.render();
    document.getElementById('stat-mood').textContent = this.pet.mood;
  },
};

if (typeof window !== 'undefined') window.BrowserMode = BrowserMode;
if (typeof module !== 'undefined') module.exports = BrowserMode;
