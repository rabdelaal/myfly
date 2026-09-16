const API = window.FLY_API || '';
let board3d, connectome;
let currentState = null;      // dernier /api/state ou réponse de /api/move
let selectedSquare = null;    // case sélectionnée (0..63)
let busy = false;             // the fly is thinking

// --- i18n : le backend parle français, la page répond en anglais ---
const I18N_RAW = {
  // humeurs
  'dort': 'sleeping', 'ravie': 'delighted', 'contente': 'content', 'grognon': 'grumpy',
  'affamée': 'starving', 'misérable': 'miserable', 'épuisée': 'exhausted', 'inconfortable': 'itchy',
  // messages d'humeur
  'Chut… elle dort profondément 💤': 'Shh… she is fast asleep 💤',
  'Elle bourdonne de bonheur autour de son terrarium 🪰✨': 'She buzzes happily around her terrarium 🪰✨',
  "Elle se lisse les antennes, l'air tranquille.": 'She smooths her antennae, calm and quiet.',
  'Elle boude dans un coin du terrarium 😤': 'She sulks in a corner of the terrarium 😤',
  'Elle tourne en rond : elle crève de faim ! 🍯': 'She paces in circles — starving! 🍯',
  "Elle a l'air vraiment triste…": 'She looks truly sad…',
  'Ses ailes traînent : elle est épuisée 💤': 'Her wings droop — she is exhausted 💤',
  'Elle se démange : une petite toilette s\u2019impose 🚿': 'She itches — a little wash is in order 🚿',
  'Elle se démange : une petite toilette s’impose 🚿': 'She itches — a little wash is in order 🚿',
  // actions
  'Elle dort déjà ! 💤': 'She is already asleep! 💤',
  'Les lumières s’éteignent… bonne nuit 🌙': 'Lights out… good night 🌙',
  'Elle est déjà réveillée !': 'She is already awake!',
  'Chut ! Elle dort. Réveille-la d’abord 💤': 'Shh! She is asleep. Wake her first 💤',
  'Elle n’a plus faim du tout et boude le sirop 🙄': 'She is completely full and snubs the syrup 🙄',
  'Elle n\u2019a plus faim du tout et boude le sirop 🙄': 'She is completely full and snubs the syrup 🙄',
  'Gloup ! Elle se jette sur le sirop, ailes vibrantes de plaisir 🍯': 'Gulp! She dives into the syrup, wings buzzing with joy 🍯',
  'Elle goûte distraitement quelques gouttes…': 'She absent-mindedly tastes a few drops…',
  'Elle frotte ses pattes avant, visiblement ravie d’être caressée 🥰': 'She rubs her front legs, visibly delighted to be petted 🥰',
  'Elle frotte ses pattes avant, visiblement ravie d\u2019être caressée 🥰': 'She rubs her front legs, visibly delighted to be petted 🥰',
  'Un petit frémissement antennaire, elle te tolère.': 'A tiny antennal shiver — she tolerates you.',
  'Pfuit ! Éclaboussée mais toute propre, elle s’essuie avec énergie 🚿': 'Splashed but sparkling, she dries off energetically 🚿',
  'Elle survit au bain avec dignité.': 'She survives the bath with dignity.',
  'Elle s’étire, déploie ses ailes et bourdonne : prête ! ☀️': 'She stretches, unfolds her wings and buzzes: ready! ☀️',
  'Elle ouvre un œil… grognonne… mais se lève.': 'She opens one eye… grumpy… but gets up.',
  'Ses lobes optiques s’affolent sur l’échiquier : elle ADORE ça ♟️': 'Her optic lobes race over the board — she LOVES this ♟️',
  'Ses lobes optiques s\u2019affolent sur l\u2019échiquier : elle ADORE ça ♟️': 'Her optic lobes race over the board — she LOVES this ♟️',
  'Elle regarde l’échiquier sans grande conviction.': 'She eyes the board without much conviction.',
  'Elle regarde l\u2019échiquier sans grande conviction.': 'She eyes the board without much conviction.',
  'Il déploie une aile et chante : parade nuptiale en cours 🎻🪰': 'He unfurls a wing and sings: courtship display in progress 🎻🪰',
  'Un petit frétillement d’aile, timide…': 'A little wing flutter, shy…',
  'Il fonce pattes en avant, ailes écartées : intimidation maximale 😠': 'He charges legs-first, wings spread: maximum intimidation 😠',
  'Il fait un pas menaçant puis hésite.': 'He takes a menacing step, then hesitates.',
  'Ses yeux composés scannent frénétiquement : elle dévore ce savoir 📚✨': 'Her compound eyes scan frantically — she devours this knowledge 📚✨',
  'Elle parcourt distraitement quelques lignes…': 'She idly skims a few lines…',
  'Elle suit la partie d’un œil vif, ailes frémissantes ♟️': 'She follows the game with a keen eye, wings aflutter ♟️',
  'La mouche dort. Réveille-la depuis l’onglet Mouche 💤': 'The fly is asleep. Wake her from the Fly tab 💤',
  // préfixe de citation study (le titre/texte cité reste tel quel)
  '📚 Elle cite « ': '📚 She recalls “ ',
  // statuts locaux FR résiduels
  'À toi — clique une pièce': 'Your turn — click a piece',
  'Trait aux noirs…': 'Black to move…',
  'Mode local : à toi (blancs), la mouche joue les noirs': 'Local mode: your move (white), the fly plays black',
  // labo : noms de groupes par clé (plus robuste que le texte)
  '__lab:mb': 'Mushroom body (MB)',
  '__lab:cx': 'Central complex (CX)',
  '__lab:dimorphic': 'Male/female dimorphic',
  '__lab:frudsx': 'fruitless / doublesex',
  '__lab:descending': 'Descending neurons',
  '__lab:optic': 'Optic lobes',
  '__lab:random4k': 'Random control (~4k)',
};
// Normalisation des apostrophes : le backend envoie du ASCII ('),
// les sources peuvent contenir des ’ typographiques — on unifie pour
// que la recherche du dictionnaire ne rate jamais (bug vu en prod).
const I18N = {};
function normApos(s) {
  return typeof s === 'string' ? s.replace(/[’‘‚‛′″]/g, "'") : s;
}
for (const [k, v] of Object.entries(I18N_RAW)) I18N[normApos(k)] = v;
function T(s) {
  if (s === null || s === undefined) return s;
  const key = normApos(s);
  if (Object.prototype.hasOwnProperty.call(I18N, key)) return I18N[key];
  const cite = normApos('📚 Elle cite « ');
  if (typeof key === 'string' && key.includes(cite)) {
    return key.replace(cite, I18N[cite]).replace(/ » : /g, '": ');
  }
  return s;
}

// --- Toasts : plus aucun échec silencieux ---
function toast(msg, type = 'info', ms = 4200) {
  const box = document.getElementById('toasts');
  if (!box) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  box.appendChild(el);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => {
    el.classList.add('out');
    setTimeout(() => el.remove(), 350);
  }, ms);
}

// fetch avec timeout (les simus cerveau prennent des dizaines de secondes)
async function api(path, body = null, timeoutMs = 180000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(API + path, {
      method: body ? 'POST' : 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : null,
      signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`Request timed out: ${path}`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// Bouton occupé : spinner + désactivé, restauration garantie
function setBusy(btn, on) {
  if (!btn) return;
  if (on) { btn.disabled = true; btn.classList.add('busy'); }
  else { btn.disabled = false; btn.classList.remove('busy'); }
}

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
  // Le status est dans l'onglet échecs (masqué par défaut) : rendre les
  // messages visibles aussi sur le panneau Tamagotchi.
  const pm = document.getElementById('pet-message');
  if (pm && pm.textContent === 'Loading…') pm.textContent = msg;
}

function isPlayersTurn() {
  return currentState && !currentState.game_over && currentState.turn === 'white' && !busy;
}

// Coups légaux depuis la case sélectionnée, au format { to, capture, promotion }
function legalMovesFrom(sq) {
  if (!currentState) return [];
  const from = FILES[sq % 8] + (Math.floor(sq / 8) + 1);
  return currentState.legal_moves
    .filter(u => u.startsWith(from))
    .map(u => {
      const to = u.slice(2, 4);
      const toSq = FILES.indexOf(to[0]) + (parseInt(to[1]) - 1) * 8;
      return { to: toSq, uci: u, capture: isCaptureUci(u) };
    });
}

// Le backend n'envoie pas le flag capture : on le déduit du FEN
function isCaptureUci(uci) {
  if (!currentState || !currentState.fen) return false;
  const to = uci.slice(2, 4);
  const toSq = FILES.indexOf(to[0]) + (parseInt(to[1]) - 1) * 8;
  return fenSquareOccupied(currentState.fen, toSq);
}

function fenSquareOccupied(fen, sq) {
  const rows = fen.split(' ')[0].split('/');
  const rank = 7 - Math.floor(sq / 8);
  const file = sq % 8;
  let col = 0;
  for (const ch of rows[rank]) {
    if (/\d/.test(ch)) { col += parseInt(ch); continue; }
    if (col === file) return true;
    col++;
  }
  return false;
}

function deselect() {
  selectedSquare = null;
  board3d.clearHighlights();
}

function onSquareClick(sq) {
  if (!isPlayersTurn()) return;

  if (sq === null) { deselect(); return; }

  if (selectedSquare !== null) {
    // Second clic : tenter le coup
    const candidates = legalMovesFrom(selectedSquare).filter(m => m.to === sq);
    if (candidates.length) {
      // Promotion : prioriser la dame
      const move = candidates.find(m => m.uci.endsWith('q')) || candidates[0];
      const from = selectedSquare;
      deselect();
      playMove(move.uci);
      return;
    }
  }

  // Sélection d'une pièce blanche
  if (fenSquareOccupied(currentState.fen, sq)) {
    const piece = fenPieceAt(currentState.fen, sq);
    if (piece && piece === piece.toUpperCase()) { // blanc = joueur
      selectedSquare = sq;
      board3d.showSelection(sq);
      board3d.showTargets(legalMovesFrom(sq));
      return;
    }
  }
  deselect();
}

function fenPieceAt(fen, sq) {
  const rows = fen.split(' ')[0].split('/');
  const rank = 7 - Math.floor(sq / 8);
  const file = sq % 8;
  let col = 0;
  for (const ch of rows[rank]) {
    if (/\d/.test(ch)) { col += parseInt(ch); continue; }
    if (col === file) return ch;
    col++;
  }
  return null;
}

async function refreshState(data) {
  if (!data) return;
  // Normalisation : /api/state dit is_game_over, /api/move dit game_over
  if (data.game_over === undefined && data.is_game_over !== undefined) {
    data.game_over = data.is_game_over;
  }
  currentState = data;
  if (data.pet) { PetUI.state = data.pet; PetUI.render(); }
  if (data.fen) board3d.setPosition(data.fen);
  deselect();
  if (data.game_over) {
    setStatus(`Game over: ${data.result || '—'}`);
  } else if (data.turn === 'black') {
    setStatus('The fly is thinking…');
  } else {
    setStatus('Your turn — click a white piece');
  }
  if (data.scores) renderCandidates(data.scores);
  // Priorité à la réaction neuronale du Tamagotchi au coup qu'elle vient de jouer
  if (data.pet_reaction && data.pet_reaction.frames && data.pet_reaction.frames.length) {
    playFrames(data.pet_reaction.frames);
  } else if (data.frames) {
    playFrames(data.frames);
  }
}

let ws = null; // WebSocket pour le streaming des spikes en temps réel
let roomInfo = null; // {code, role, ...} quand on est dans un salon
let liveWS = null; // flux continu du cerveau (toggle ⚡ Live)
let liveWant = false; // l'utilisateur veut le live (reconnexion auto)
let liveRetries = 0;
let audioCtx = null; // clics Geiger des spikes (toggle 🔊 Son)
let soundOn = false;

function spikeClick(t) {
  const o = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  o.type = 'square';
  o.frequency.value = 1700 + Math.random() * 1600;
  g.gain.setValueAtTime(0.05, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.03);
  o.connect(g).connect(audioCtx.destination);
  o.start(t);
  o.stop(t + 0.035);
}

function playSpikeSound(nSpikes) {
  if (!soundOn || !audioCtx) return;
  const now = audioCtx.currentTime;
  const clicks = Math.min(nSpikes, 8);
  for (let i = 0; i < clicks; i++) spikeClick(now + Math.random() * 0.09);
}

function toggleSound() {
  const btn = document.getElementById('btn-sound');
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  soundOn = !soundOn;
  btn.classList.toggle('on', soundOn);
  btn.setAttribute('aria-pressed', String(soundOn));
  toast(soundOn ? '🔊 Spike clicks on' : '🔇 Sound off', 'info', 1800);
}

function toggleLive() {
  const btn = document.getElementById('btn-live');
  if (liveWS || liveWant) {
    liveWant = false;
    liveRetries = 0;
    if (liveWS) liveWS.close();
    liveWS = null;
    btn.textContent = '⚡ Live';
    btn.classList.remove('live-on');
    btn.setAttribute('aria-pressed', 'false');
    document.getElementById('hud-frames').style.display = 'none';
    return;
  }
  liveWant = true;
  liveRetries = 0;
  openLive();
}

function openLive() {
  const btn = document.getElementById('btn-live');
  if (!liveWant) return;
  const proto = (API.replace(/^http/, 'ws')) + '/ws/live';
  try {
    liveWS = new WebSocket(proto);
  } catch (e) {
    scheduleLiveRetry(btn);
    return;
  }
  liveWS.onopen = () => {
    liveRetries = 0;
    btn.textContent = '⏸ Live ON';
    btn.classList.add('live-on');
    btn.setAttribute('aria-pressed', 'true');
    document.getElementById('hud-frames').style.display = '';
  };
  liveWS.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'live' && connectome) {
      connectome.setActivation(m.spikes);
      playSpikeSound(m.spikes.length);
      const n = document.getElementById('hud-spikes');
      if (n) n.textContent = m.spikes.length;
      document.getElementById('stat-mood').textContent = T(
        (m.sleeping ? '💤 ' : '') + m.mood);
    }
  };
  liveWS.onclose = () => {
    liveWS = null;
    if (liveWant) scheduleLiveRetry(btn);
  };
  liveWS.onerror = () => { try { liveWS.close(); } catch (e) { /* retry via onclose */ } };
}

function scheduleLiveRetry(btn) {
  if (!liveWant) return;
  liveRetries++;
  if (liveRetries > 6) {
    liveWant = false;
    btn.textContent = '⚡ Live';
    btn.classList.remove('live-on');
    toast('Live feed unavailable — backend restarting?', 'err');
    document.getElementById('hud-frames').style.display = 'none';
    return;
  }
  btn.textContent = `⏳ Live…(${liveRetries})`;
  setTimeout(openLive, Math.min(2000 * liveRetries, 10000));
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  } else {
    setStatus('Not connected to backend — reload the page');
    toast('Not connected — reload the page', 'err');
  }
}

let wsRetries = 0;
function connectWS() {
  try {
    ws = new WebSocket((API.replace(/^http/, 'ws')) + '/ws');
    ws.onopen = () => { wsRetries = 0; };
    ws.onmessage = async (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === 'frame' && connectome) {
        connectome.setActivation(m.frame); // spikes streamés pendant la réflexion
      } else if (m.type === 'result') {
        busy = false;
        refreshState(m.data);
        if (m.data.pet_reaction && m.data.pet_reaction.message) {
          document.getElementById('pet-message').textContent = T(m.data.pet_reaction.message);
        }
        if (currentState && !currentState.game_over && currentState.turn === 'white') {
          setStatus('Your turn — click a white piece');
        }
      } else if (m.type === 'error') {
        busy = false;
        setStatus(`Error: ${m.detail}`);
        toast(`Move error: ${m.detail}`, 'err');
      } else if (m.type === 'room') {
        roomInfo = m.data;
        // Le plateau suit le salon (premier affichage + coups adverses)
        if (m.data.fen) await refreshState({ fen: m.data.fen, turn: m.data.turn });
        renderRoom();
      }
    };
    ws.onclose = () => {
      ws = null;
      // Reconnexion silencieuse : le REST prend le relais entre-temps
      if (wsRetries < 5) {
        wsRetries++;
        setTimeout(() => { if (!ws) connectWS(); }, 2500 * wsRetries);
      }
    };
  } catch (e) { ws = null; }
}

async function playMove(uci) {
  if (busy) return;
  if (window.LOCAL_MODE && typeof BrowserMode !== 'undefined') {
    return BrowserMode.playMove(uci);
  }
  busy = true;
  setStatus('The fly is thinking…');
  renderCandidates(null); // placeholder "computing"
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'move', move: uci }));
    return; // la réponse arrive via ws.onmessage
  }
  try {
    const data = await api('/api/move', { move: uci });
    await refreshState(data);
    if (data.pet_reaction && data.pet_reaction.message) {
      PetUI.lastActionMessage = T(data.pet_reaction.message);
      document.getElementById('pet-message').textContent = T(data.pet_reaction.message);
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    toast(`Move failed: ${err.message}`, 'err');
  } finally {
    busy = false;
    if (currentState && !currentState.game_over && currentState.turn === 'white') {
      setStatus('Your turn — click a white piece');
    }
  }
}

function renderRoom() {
  const el = document.getElementById('room-status');
  const leaveBtn = document.getElementById('btn-room-leave');
  if (!roomInfo) {
    el.textContent = '';
    leaveBtn.classList.add('hidden');
    return;
  }
  leaveBtn.classList.remove('hidden');
  const r = roomInfo;
  const role = r.role === 'spec' ? 'spectator' : `you play ${r.role === 'white' ? 'white' : 'black'}`;
  el.textContent = `Room ${r.code} · ${role} · ♔${r.white} ♚${r.black}` +
    (r.spectators ? ` · 👁 ${r.spectators}` : '') +
    (r.game_over ? ' · over' : ` · ${r.turn === 'white' ? 'white' : 'black'} to move`);
  const mine = (r.turn === 'white' && r.role === 'white') || (r.turn === 'black' && r.role === 'black');
  setStatus(r.game_over ? `Game over — room ${r.code}` :
    mine ? `Your turn (${r.code}) — click a piece` :
    r.role === 'spec' ? `Watching (${r.code})` : `Opponent moving (${r.code})…`);
}

function renderCandidates(scores) {
  const el = document.getElementById('candidates');
  if (!scores) {
    el.innerHTML = '<span class="candidates-hint">🧠 fly is evaluating…</span>';
    return;
  }
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (!entries.length) {
    el.innerHTML = '<span class="candidates-hint">No candidate moves.</span>';
    return;
  }
  const max = Math.max(...entries.map(e => Math.abs(e[1])), 1e-6);
  el.innerHTML = entries.map(([move, score], i) => `
    <div class="candidate">
      <span class="rank">${i + 1}</span>
      <span class="mv">${move}</span>
      <div class="bar"><div style="width:${(Math.abs(score) / max) * 100}%"></div></div>
      <span class="sc">${score.toFixed(3)}</span>
    </div>
  `).join('');
}

let frameTimer = null;
function playFrames(frames) {
  if (!connectome) return;
  if (frameTimer) clearInterval(frameTimer);
  const hud = document.getElementById('hud-frames');
  const n = document.getElementById('hud-spikes');
  if (hud) hud.style.display = '';
  let i = 0;
  frameTimer = setInterval(() => {
    if (i >= frames.length) {
      clearInterval(frameTimer);
      frameTimer = null;
      if (hud) hud.style.display = 'none';
      return;
    }
    connectome.setActivation(frames[i]);
    if (n) n.textContent = frames[i].length;
    playSpikeSound(frames[i].length); // Geiger local : marche aussi sur les frames du worker (mode local)
    i++;
  }, 26);
}

function switchTab(which) {
  const tabs = ['pet', 'chess', 'lab', 'sigils', 'mind'];
  for (const t of tabs) {
    const btn = document.getElementById('tab-' + t);
    if (btn) {
      btn.classList.toggle('active', t === which);
      btn.setAttribute('aria-selected', String(t === which));
    }
  }
  document.getElementById('pet-view').classList.toggle('hidden', which !== 'pet');
  document.getElementById('chess-view').classList.toggle('hidden', which !== 'chess');
  document.getElementById('lab-view').classList.toggle('hidden', which !== 'lab');
  document.getElementById('sigils-view').classList.toggle('hidden', which !== 'sigils');
  document.getElementById('mind-view').classList.toggle('hidden', which !== 'mind');
  // ⚗️ Labo : si le premier chargement a échoué (backend occupé au démarrage),
  // retenter à chaque ouverture de l'onglet.
  if (which === 'lab' && window.LabUI) LabUI.refresh().catch(() => {});
  if (which === 'mind' && window.MindUI) MindUI.refresh().catch(() => {});
  if (which === 'chess' && board3d) board3d.resize(); // le canvas était masqué (taille 0)
}

function setBackendPill(mode, text) {
  const pill = document.getElementById('backend-status');
  const label = document.getElementById('backend-status-text');
  if (!pill || !label) return;
  pill.classList.remove('online', 'local', 'offline');
  pill.classList.add(mode);
  label.textContent = text;
}

function renderBrainChips(info) {
  const el = document.getElementById('brain-chips');
  if (!el) return;
  const fmt = (n) => Number(n).toLocaleString('en-US');
  el.innerHTML = [
    `🧠 <b>${info.n_neurons > 100000 ? 'MaleCNS REAL' : 'synthetic demo'}</b>`,
    `${fmt(info.n_neurons)} neurons`,
    `${fmt(info.n_synapses)} synapses`,
    info.synapse_model === 'alpha'
      ? `α-Shiu ${info.wsyn_mv?.toFixed(3) ?? '?'} mV${info.n_sugar_grn ? ` · ${info.n_sugar_grn} sweet GRNs` : ''}`
      : 'current synapses',
    info.readout_trained ? '✓ trained readout' : '○ untrained readout',
  ].map(c => `<span>${c}</span>`).join('');
}

function countUp(el, target, dur = 900) {
  if (!el) return;
  const t0 = performance.now();
  const fmt = (n) => Math.round(n).toLocaleString('en-US');
  function tick(t) {
    const k = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(target * e);
    if (k < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

async function waitForLibs() {
  // Les fallbacks CDN sont async : attendre THREE + Chess (8 s max),
  // sinon message fatal explicite au lieu d'un écran vide.
  const t0 = Date.now();
  while (!(window.THREE && window.Chess)) {
    if (Date.now() - t0 > 8000) {
      throw new Error('3D/chess libraries failed to load (offline? vendor/ missing?)');
    }
    await new Promise(r => setTimeout(r, 100));
  }
}

async function init() {
  await waitForLibs();
  // Choisir l'onglet AVANT de créer les canvas (un canvas dans un onglet
  // masqué a clientWidth = 0 et ne se dessine jamais)
  const params = new URLSearchParams(location.search);
  if (['chess', 'lab', 'sigils', 'mind'].includes(params.get('tab'))) {
    switchTab(params.get('tab'));
  }

  const boardCanvas = document.getElementById('board-canvas');
  const connCanvas = document.getElementById('connectome-canvas');

  boardCanvas.width = boardCanvas.clientWidth;
  boardCanvas.height = boardCanvas.clientHeight;
  connCanvas.width = connCanvas.clientWidth * window.devicePixelRatio;
  connCanvas.height = connCanvas.clientHeight * window.devicePixelRatio;

  board3d = new Chess3D(boardCanvas, onSquareClick);

  // Sans backend : bascule 100 % navigateur (GitHub Pages) au lieu de mourir
  try {
    await Promise.race([
      fetch(API + '/api/info').then((r) => { if (!r.ok) throw new Error('pas de backend'); }),
      new Promise((_, rej) => setTimeout(() => rej(new Error('timeout backend')), 2500)),
    ]);
  } catch (e) {
    if (typeof BrowserMode !== 'undefined' && typeof Chess !== 'undefined') {
      setBackendPill('local', 'local demo brain');
      await BrowserMode.start({ connCanvas });
      // Même en local : les onglets Labo/Sigils/Mind ont leurs replis locaux
      // (sinon ils restent vides sur GitHub Pages — init() faisait return avant).
      LabUI.init().catch(e => console.warn('Labo indisponible', e));
      Sigils.init().catch(e => console.warn('Sigils indisponibles', e));
      if (window.MindUI) MindUI.init().catch(e => console.warn('Mind indisponible', e));
      return;
    }
    setBackendPill('offline', 'no backend');
    document.getElementById('pet-message').textContent =
      'No backend (serve this page with chess.js + browser.js, or launch uvicorn)';
    toast('Backend unreachable — running without live brain', 'err');
    return;
  }
  setBackendPill('online', 'live brain');

  let info;
  try {
    info = await api('/api/info', null, 15000);
  } catch (e) {
    setBackendPill('offline', 'info failed');
    toast(`Backend info failed: ${e.message}`, 'err');
    throw e;
  }
  countUp(document.getElementById('stat-neurons'), info.n_neurons);
  countUp(document.getElementById('stat-synapses'), info.n_synapses);
  // Bandeau d'état : on voit immédiatement sur quel cerveau on joue
  renderBrainChips(info);
  document.getElementById('backend-status-text').textContent =
    info.n_neurons > 100000 ? 'live brain' : 'synthetic brain';
  // Fiche dimorphisme (Cell 2026, Table 1) : chiffres + part des hotspots fru/dsx
  if (info.n_hotspot) {
    const card = document.getElementById('dimorphism-card');
    card.hidden = false;
    card.innerHTML =
      `♂ <b>${info.n_male_specific.toLocaleString('en-US')}</b> male-specific · ` +
      `◐ <b>${info.n_hotspot.toLocaleString('en-US')}</b> hotspots · ` +
      `fru+ <b>${info.n_fru.toLocaleString('en-US')}</b> · ` +
      `dsx+ <b>${info.n_dsx.toLocaleString('en-US')}</b> ` +
      `<span title="90.4 % of male-specific neurons are fru+/dsx+; the “potentially” class stays separate">ⓘ</span>`;
  }

  // Vraies positions 3D du connectome (retombe sur du hasard si indispo)
  let positions = null, categories = null, edges = null, hotspot = null;
  try {
    const conn = await api('/api/connectome', null, 60000);
    positions = conn.positions;
    categories = conn.category;
    edges = conn.edges;
    hotspot = conn.hotspot || null;
    window.CONNECTOME_DATA = { positions, category: categories, edges, hotspot };
  } catch (e) { console.warn('Positions du connectome indisponibles', e); }

  try {
    connectome = new ConnectomeView(
      connCanvas, Math.min(info.n_neurons, positions ? positions.length : 5000),
      positions, categories, hotspot
    );
  } catch (e) {
    document.getElementById('gl-status').textContent = '⚠️ ' + e.message;
    toast('3D brain unavailable: ' + e.message, 'err', 8000);
    document.getElementById('btn-theater').disabled = true;
    connectome = null;
  }
  if (!connectome) {
    document.getElementById('btn-live').disabled = true;
    document.getElementById('btn-sound').disabled = true;
  }
  if (edges && edges.a && connectome) connectome.setEdges(edges);
  if (connectome && connectome._glErrors && connectome._glErrors.length) {
    document.getElementById('gl-status').textContent =
      '⚠️ WebGL : ' + connectome._glErrors[0].slice(0, 120);
    toast('WebGL warning: ' + connectome._glErrors[0].slice(0, 100), 'err');
  } else {
    document.getElementById('gl-status').textContent = '· WebGL OK';
  }

  document.getElementById('btn-theater').onclick = () => Theater.open();
  document.getElementById('btn-live').onclick = toggleLive;
  document.getElementById('btn-sound').onclick = toggleSound;

  const state = await api('/api/state');
  await refreshState(state);

  const bindBusy = (id, fn) => {
    const btn = document.getElementById(id);
    btn.onclick = async () => {
      setBusy(btn, true);
      try { await fn(); }
      catch (e) { setStatus(`Error: ${e.message}`); toast(`${id}: ${e.message}`, 'err'); }
      finally { setBusy(btn, false); }
    };
  };
  bindBusy('btn-new', async () => {
    await api('/api/new', { player_color: 'white' });
    renderCandidates(null); // efface les coups de l'ancienne partie
    const s = await api('/api/state');
    await refreshState(s);
    toast('✨ New game — your move', 'ok', 2200);
  });
  bindBusy('btn-undo', async () => {
    await api('/api/undo', {});
    deselect();
    const s = await api('/api/state');
    await refreshState(s);
  });
  bindBusy('btn-hint', async () => {
    setStatus('The fly is computing a hint…');
    const d = await api('/api/hint');
    if (d.hint) { setStatus(`Hint: ${d.hint}`); toast(`💡 Hint: ${d.hint}`, 'ok'); }
    if (d.scores) renderCandidates(d.scores);
  });
  document.getElementById('btn-room-create').onclick = () => wsSend({ type: 'create', fly: true });
  document.getElementById('btn-room-create-h2h').onclick = () => wsSend({ type: 'create', fly: false });
  document.getElementById('btn-room-join').onclick = () => {
    const code = document.getElementById('room-code').value.trim().toUpperCase();
    if (!/^[A-Z0-9]{4}$/.test(code)) { toast('Room code: 4 letters/digits', 'err'); return; }
    wsSend({ type: 'join', code });
  };
  document.getElementById('btn-room-leave').onclick = () => {
    wsSend({ type: 'leave' });
    roomInfo = null;
    renderRoom();
  };

  window.addEventListener('resize', () => {
    board3d.resize();
    if (connectome) connectome.resize();
  });

  // Onglets + Tamagotchi + Labo + WebSocket temps réel
  document.getElementById('tab-pet').onclick = () => switchTab('pet');
  document.getElementById('tab-chess').onclick = () => switchTab('chess');
  document.getElementById('tab-lab').onclick = () => switchTab('lab');
  document.getElementById('tab-sigils').onclick = () => switchTab('sigils');
  document.getElementById('tab-mind').onclick = () => switchTab('mind');
  await PetUI.init();
  // Pas d'await : un /api lent au démarrage ne doit pas retarder le WS
  LabUI.init().catch(e => console.warn('Labo indisponible', e));
  Sigils.init().catch(e => console.warn('Sigils indisponibles', e));
  if (window.MindUI) MindUI.init().catch(e => console.warn('Mind indisponible', e));
  connectWS();
  if (new URLSearchParams(location.search).has('theater')) Theater.open();
}

init().catch(err => {
  console.error(err);
  const msg = `Error: ${err.message} — is the backend running on ${API || 'this server'} ?`;
  setStatus(msg);
  const pm = document.getElementById('pet-message');
  if (pm) pm.textContent = msg;
  toast(msg, 'err', 8000);
});
