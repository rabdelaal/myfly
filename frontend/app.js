const API = window.FLY_API || '';
let board3d, connectome;
let currentState = null;      // dernier /api/state ou réponse de /api/move
let selectedSquare = null;    // case sélectionnée (0..63)
let busy = false;             // the fly is thinking

async function api(path, body = null) {
  const r = await fetch(API + path, {
    method: body ? 'POST' : 'GET',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : null,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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
}

function toggleLive() {
  const btn = document.getElementById('btn-live');
  if (liveWS) {
    liveWS.close();
    liveWS = null;
    btn.textContent = '⚡ Live';
    btn.classList.remove('live-on');
    return;
  }
  const proto = (API.replace(/^http/, 'ws')) + '/ws/live';
  liveWS = new WebSocket(proto);
  liveWS.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'live' && connectome) {
      connectome.setActivation(m.spikes);
      playSpikeSound(m.spikes.length);
      document.getElementById('stat-mood').textContent =
        (m.sleeping ? '💤 ' : '') + m.mood;
    }
  };
  liveWS.onclose = () => {
    liveWS = null;
    btn.textContent = '⚡ Live';
    btn.classList.remove('live-on');
  };
  btn.textContent = '⏸ Live ON';
  btn.classList.add('live-on');
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  } else {
    setStatus('Not connected to backend — reload the page');
  }
}

function connectWS() {  try {
    ws = new WebSocket((API.replace(/^http/, 'ws')) + '/ws');
    ws.onmessage = async (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === 'frame' && connectome) {
        connectome.setActivation(m.frame); // spikes streamés pendant la réflexion
      } else if (m.type === 'result') {
        busy = false;
        refreshState(m.data);
        if (m.data.pet_reaction && m.data.pet_reaction.message) {
          document.getElementById('pet-message').textContent = m.data.pet_reaction.message;
        }
        if (currentState && !currentState.game_over && currentState.turn === 'white') {
          setStatus('Your turn — click a white piece');
        }
      } else if (m.type === 'error') {
        busy = false;
        setStatus(`Error: ${m.detail}`);
      } else if (m.type === 'room') {
        roomInfo = m.data;
        // Le plateau suit le salon (premier affichage + coups adverses)
        if (m.data.fen) await refreshState({ fen: m.data.fen, turn: m.data.turn });
        renderRoom();
      }
    };
    ws.onclose = () => { ws = null; };
  } catch (e) { ws = null; }
}

async function playMove(uci) {
  if (busy) return;
  if (window.LOCAL_MODE && typeof BrowserMode !== 'undefined') {
    return BrowserMode.playMove(uci);
  }
  busy = true;
  setStatus('The fly is thinking…');
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'move', move: uci }));
    return; // la réponse arrive via ws.onmessage
  }
  try {
    const data = await api('/api/move', { move: uci });
    await refreshState(data);
    if (data.pet_reaction && data.pet_reaction.message) {
      PetUI.lastActionMessage = data.pet_reaction.message;
      document.getElementById('pet-message').textContent = data.pet_reaction.message;
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`);
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

function renderCandidates(scores) {  const el = document.getElementById('candidates');
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const max = Math.max(...entries.map(e => Math.abs(e[1])), 1e-6);
  el.innerHTML = entries.map(([move, score]) => `
    <div class="candidate">
      <span style="width:60px">${move}</span>
      <div class="bar"><div style="width:${(Math.abs(score) / max) * 100}%"></div></div>
      <span style="width:50px;text-align:right">${score.toFixed(3)}</span>
    </div>
  `).join('');
}

let frameTimer = null;
function playFrames(frames) {
  if (frameTimer) clearInterval(frameTimer);
  let i = 0;
  frameTimer = setInterval(() => {
    if (i >= frames.length) { clearInterval(frameTimer); return; }
    connectome.setActivation(frames[i]);
    i++;
  }, 26);
}

function switchTab(which) {
  const tabs = ['pet', 'chess', 'lab', 'sigils'];
  for (const t of tabs) {
    const btn = document.getElementById('tab-' + t);
    if (btn) btn.classList.toggle('active', t === which);
  }
  document.getElementById('pet-view').classList.toggle('hidden', which !== 'pet');
  document.getElementById('chess-view').classList.toggle('hidden', which !== 'chess');
  document.getElementById('lab-view').classList.toggle('hidden', which !== 'lab');
  document.getElementById('sigils-view').classList.toggle('hidden', which !== 'sigils');
  // ⚗️ Labo : si le premier chargement a échoué (backend occupé au démarrage),
  // retenter à chaque ouverture de l'onglet.
  if (which === 'lab' && window.LabUI) LabUI.refresh().catch(() => {});
  if (which === 'chess' && board3d) board3d.resize(); // le canvas était masqué (taille 0)
}

async function init() {
  // Choisir l'onglet AVANT de créer les canvas (un canvas dans un onglet
  // masqué a clientWidth = 0 et ne se dessine jamais)
  const params = new URLSearchParams(location.search);
  if (params.get('tab') === 'chess') switchTab('chess');

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
      await BrowserMode.start({ connCanvas });
      return;
    }
    document.getElementById('backend-status').textContent =
      'Pas de backend (lance uvicorn ou sers cette page avec chess.js + browser.js)';
  }

  const info = await api('/api/info');
  document.getElementById('stat-neurons').textContent = info.n_neurons.toLocaleString();
  document.getElementById('stat-synapses').textContent = info.n_synapses.toLocaleString();
  // Bandeau d'état : on voit immédiatement sur quel cerveau on joue
  const brainKind = info.n_neurons > 100000
    ? 'cerveau RÉEL MaleCNS v1.0'
    : 'cerveau synthétique de démonstration';
  const synDesc = info.synapse_model === 'alpha'
    ? `α-synapses Shiu (Wsyn ${info.wsyn_mv?.toFixed(3) ?? '?'} mV${info.n_sugar_grn ? `, ${info.n_sugar_grn} GRN sucrées` : ''})`
    : 'synapses courants';
  document.getElementById('backend-status').textContent =
    `${brainKind} · ${info.n_neurons.toLocaleString('fr-FR')} neurones · ` +
    `${info.n_synapses.toLocaleString('fr-FR')} synapses · ${synDesc} · ` +
    (info.readout_trained ? 'readout entraîné' : 'readout non entraîné') +
    (info.readout_looped ? ' · looped dispo' : '') +
    ` · encodeur ${info.encoder ?? 'classic'} · DN ${info.dn_mode ?? 'all'}`;
  // Fiche dimorphisme (Cell 2026, Table 1) : chiffres + part des hotspots fru/dsx
  if (info.n_hotspot) {
    const card = document.getElementById('dimorphism-card');
    card.hidden = false;
    card.innerHTML =
      `♂ <b>${info.n_male_specific.toLocaleString('fr-FR')}</b> male-specific · ` +
      `◐ <b>${info.n_hotspot.toLocaleString('fr-FR')}</b> hotspots · ` +
      `fru+ <b>${info.n_fru.toLocaleString('fr-FR')}</b> · ` +
      `dsx+ <b>${info.n_dsx.toLocaleString('fr-FR')}</b> ` +
      `<span title="90,4 % des male-specific sont fru+/dsx+ ; les « potentially » restent une catégorie à part">ⓘ</span>`;
  }

  // Vraies positions 3D du connectome (retombe sur du hasard si indispo)
  let positions = null, categories = null, edges = null, hotspot = null;
  try {
    const conn = await api('/api/connectome');
    positions = conn.positions;
    categories = conn.category;
    edges = conn.edges;
    hotspot = conn.hotspot || null;
    window.CONNECTOME_DATA = { positions, category: categories, edges, hotspot };
  } catch (e) { console.warn('Positions du connectome indisponibles', e); }

  connectome = new ConnectomeView(
    connCanvas, Math.min(info.n_neurons, positions ? positions.length : 5000),
    positions, categories, hotspot
  );
  if (edges && edges.a) connectome.setEdges(edges);
  if (connectome._glErrors && connectome._glErrors.length) {
    document.getElementById('gl-status').textContent =
      '⚠️ WebGL : ' + connectome._glErrors[0].slice(0, 120);
  } else {
    document.getElementById('gl-status').textContent = '· WebGL OK';
  }

  document.getElementById('btn-theater').onclick = () => Theater.open();
  document.getElementById('btn-live').onclick = toggleLive;
  document.getElementById('btn-sound').onclick = toggleSound;

  const state = await api('/api/state');
  await refreshState(state);

  document.getElementById('btn-new').onclick = async () => {
    const d = await api('/api/new', { player_color: 'white' });
    const s = await api('/api/state');
    await refreshState(s);
  };
  document.getElementById('btn-undo').onclick = async () => {
    const d = await api('/api/undo', {});
    deselect();
    const s = await api('/api/state');
    await refreshState(s);
  };
  document.getElementById('btn-hint').onclick = async () => {
    setStatus('The fly is computing a hint…');
    const d = await api('/api/hint');
    if (d.hint) setStatus(`Hint: ${d.hint}`);
    if (d.scores) renderCandidates(d.scores);
  };
  document.getElementById('btn-room-create').onclick = () => wsSend({ type: 'create', fly: true });
  document.getElementById('btn-room-create-h2h').onclick = () => wsSend({ type: 'create', fly: false });
  document.getElementById('btn-room-join').onclick = () => {
    const code = document.getElementById('room-code').value.trim().toUpperCase();
    if (code) wsSend({ type: 'join', code });
  };
  document.getElementById('btn-room-leave').onclick = () => {
    wsSend({ type: 'leave' });
    roomInfo = null;
    renderRoom();
  };

  window.addEventListener('resize', () => {
    board3d.resize();
    connectome.resize();
  });

  // Onglets + Tamagotchi + Labo + WebSocket temps réel
  document.getElementById('tab-pet').onclick = () => switchTab('pet');
  document.getElementById('tab-chess').onclick = () => switchTab('chess');
  document.getElementById('tab-lab').onclick = () => switchTab('lab');
  document.getElementById('tab-sigils').onclick = () => switchTab('sigils');
  await PetUI.init();
  // Pas d'await : un /api lent au démarrage ne doit pas retarder le WS
  LabUI.init().catch(e => console.warn('Labo indisponible', e));
  Sigils.init().catch(e => console.warn('Sigils indisponibles', e));
  connectWS();
  if (new URLSearchParams(location.search).has('theater')) Theater.open();
}

init().catch(err => {
  console.error(err);
  const msg = `Error: ${err.message} — is the backend running on ${API || 'this server'} ?`;
  setStatus(msg);
  const pm = document.getElementById('pet-message');
  if (pm) pm.textContent = msg;
});
