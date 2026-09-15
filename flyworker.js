// Cerveau synthétique 100 % navigateur (mode démo Pages, sans backend).
// Miroir de backend/data_loader._synthetic_connectome + brain FlyBrain
// (mode "current") + encoding.BoardEncoder, en Float32Array/CSC.
// Différences assumées vs backend : leak exact via Spear.fastExp (Euler côté
// Python — écart O(dt²), documenté), readout linéaire NON entraîné (comme un
// backend frais), 5000 neurones (le vrai MaleCNS reste côté serveur).
// Testable sous node : `node -e "const w=require('./flyworker.js'); w.selftest()"`
'use strict';

try {
  if (typeof importScripts === 'function') importScripts('spear.js');
} catch (e) { /* repli local (ex. test node) */ }

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRandn(seed) {
  const rnd = mulberry32(seed);
  let spare = null;
  return function () {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    const m = Math.sqrt(-2 * Math.log(u));
    spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };
}

const CFG = { n: 5000, dens: 0.005, tau: 20, dt: 1, vth: -45, vrest: -52, vreset: -52, tausyn: 5, refr: 3, delay: 2, gain: 20, norm: 450, nf: 788, stimP: 0.01, wsyn: 1.2 };
// Wsyn calibré (node calibrate()) : 0.3→0.1Hz, 0.5→0.3, 0.8→0.8, 1.2→1.2Hz,
// 2.0→2.4Hz moteurs (sensoriels ~10Hz stables, pas d'avalanche). Le 0.164 de
// Shiu vaut pour le vrai MaleCNS (degré 141, comptes ×2591), pas ici.
const P_INT = { 1: 0.5, 2: 0.8, 3: 0.8, 4: 1.0, 5: 1.5, 6: 2.0 }; // pawn..king
const CHANNELS = ['sweet', 'touch', 'smell', 'water', 'light', 'vibration', 'pattern', 'frustration'];
const RECIPES = {
  feed: { sweet: 1, smell: 0.8, touch: 0.3 }, pet: { touch: 1, smell: 0.2 },
  clean: { water: 1, touch: 0.6 }, wake: { light: 1, vibration: 0.5 },
  chess: { pattern: 1, frustration: 0.4 },
  courtship: { smell: 1, vibration: 0.6, touch: 0.3 },
  threat: { touch: 1, vibration: 1, frustration: 0.5 },
  study: { pattern: 1, light: 0.5, touch: 0.2 },
};

function buildNet(n, seed) {
  const randn = makeRandn(seed), rnd = mulberry32(seed ^ 0x9e37);
  const ns = Math.floor(0.3 * n), nm = Math.floor(0.2 * n);
  const ni = n - ns - nm, descN = Math.min(100, ni);
  const isSens = i => i < ns, isMot = i => i >= ns + ni, isDesc = i => i >= ns + ni - descN && i < ns + ni;
  // Paires (pré, post) : on jette sensoriel→sensoriel comme le backend
  const cols = [], data = [];
  const target = Math.floor(n * n * CFG.dens);
  let sumAbs = 0;
  const rows = [];
  // Voies feedforward (le hasard pur ne propage pas : 25 entrées × 30 %
  // de sensoriels actifs × |w| < Vth — mesuré silencieux même ×5).
  // S→I et S/I→Moteurs systématiques, fond récurrent à 25 % : le pattern
  // d'entrée pilote les moteurs tout en gardant du mélange.
  for (let k = 0; k < target; k++) {
    const pre = (rnd() * n) | 0, post = (rnd() * n) | 0;
    if (pre < ns && post < ns) continue;
    const preSens = pre < ns, postMot = post >= ns + ni, postInter = post >= ns && post < ns + ni;
    let keep = 0.25;
    if (preSens && (postInter || postMot)) keep = 1.0;
    else if (!preSens && postMot) keep = 0.6;
    if (rnd() > keep) continue;
    const s = rnd() < 0.8 ? 1 : -1;
    // Loi lourde (MaleCNS : médiane 2, max 2591) : 5 % de synapses fortes
    // 20-100× qui portent la propagation, fond faible qui mélange. Sans
    // queue lourde le fan-in moyen (~0.3) reste sous Vth=1 — mesuré silencieux.
    let c = 1 + ((rnd() * 3) | 0);
    if (rnd() < 0.05) c = 20 + ((rnd() * 80) | 0);
    const w = -(Math.log(1 - rnd()) + Math.log(1 - rnd())) * s * c; // gamma(2,1)×compte
    rows.push(pre, post);
    data.push(w);
    sumAbs += Math.abs(w);
  }
  const scale = 1 / (sumAbs / data.length); // poids BRUTS signés × compte
  // (régime alpha-lite comme Shiu : w = compte × signe × Wsyn ; les kicks
  // sont en mV absolus, la queue lourde (5 % à 20-100) franchit le seuil de
  // 7 mV là où le LIF-current ne vivait que de la moyenne — mesuré silencieux)
  for (let k = 0; k < data.length; k++) data[k] *= scale;
  // CSR (lignes = post)
  const cnt = new Int32Array(n);
  for (let k = 0; k < data.length; k++) cnt[rows[2 * k + 1]]++;
  const indptr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) indptr[i + 1] = indptr[i] + cnt[i];
  const indices = new Int32Array(data.length), vals = new Float32Array(data.length);
  const fill = Int32Array.from(indptr.subarray(0, n));
  for (let k = 0; k < data.length; k++) {
    const p = fill[rows[2 * k + 1]]++;
    indices[p] = rows[2 * k];
    vals[p] = data[k];
  }
  // CSC (colonnes = pré) pour le scatter sur neurones actifs uniquement
  const cCnt = new Int32Array(n);
  for (let k = 0; k < data.length; k++) cCnt[rows[2 * k]]++;
  const colptr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) colptr[i + 1] = colptr[i] + cCnt[i];
  const rowidx = new Int32Array(data.length), cvals = new Float32Array(data.length);
  const cfill = Int32Array.from(colptr.subarray(0, n));
  for (let k = 0; k < data.length; k++) {
    const p = cfill[rows[2 * k]]++;
    rowidx[p] = rows[2 * k + 1];
    cvals[p] = data[k];
  }
  return { n, ns, nm, indptr, indices, vals, colptr, rowidx, cvals, isSens, isMot, isDesc, nnz: data.length };
}

function buildEncoder(ns, seed) {
  const randn = makeRandn(seed), nf = CFG.nf;
  const proj = new Float32Array(nf * ns); // [i*ns+j]
  const s = 1 / Math.sqrt(nf);
  for (let k = 0; k < proj.length; k++) proj[k] = randn() * s;
  return { ns, nf, proj };
}

// f = {pieces:[[sq,pt,col]...], legal, check, mate, stale, turn, fullmove, material}
function features(f, nf) {
  const v = new Float32Array(nf);
  for (const [sq, pt, col] of f.pieces) v[sq * 12 + (pt - 1) * 2 + (col ? 0 : 1)] = P_INT[pt] || 0;
  const o = 768;
  v[o] = f.legal / 40; v[o + 1] = f.check ? 1 : 0; v[o + 2] = f.mate ? 1 : 0;
  v[o + 3] = f.stale ? 1 : 0; v[o + 4] = f.turn ? 1 : 0; v[o + 5] = f.fullmove / 100;
  v[o + 6] = f.material / 40;
  return v;
}

function encodeBatch(enc, featsList) {
  const B = featsList.length, out = new Float32Array(enc.ns * B);
  for (let b = 0; b < B; b++) {
    const f = featsList[b], col = out.subarray(b * enc.ns, (b + 1) * enc.ns);
    for (let j = 0; j < enc.ns; j++) {
      let s = 0;
      for (let i = 0; i < enc.nf; i++) s += f[i] * enc.proj[i * enc.ns + j];
      col[j] = s * CFG.gain;
    }
    let nrm = 0;
    for (let j = 0; j < enc.ns; j++) nrm += col[j] * col[j];
    nrm = Math.sqrt(nrm) || 1;
    const k = CFG.norm / nrm;
    for (let j = 0; j < enc.ns; j++) col[j] *= k;
  }
  return out; // [b*ns+j]
}

const fastExp = (typeof Spear !== 'undefined' && Spear.fastExp)
  || ((x) => Math.exp(-x)); // repli exact hors worker/navigateur

function simulate(net, currents, B, steps, recordEvery, collectFrames, wsyn) {
  // Alpha-lite (miroir de brain.py, modèle Shiu) : le LIF-current restait
  // silencieux car il ne vit que de la moyenne ; ici kicks mV absolus +
  // drive Poisson, régime prouvé vivant sur le vrai connectome.
  // V en mV (-52/-45), g en mV (tau_syn=5), délai 2 pas, réfractaire 3 pas.
  const { n, ns, colptr, rowidx, cvals } = net;
  const W = wsyn === undefined ? CFG.wsyn : wsyn;
  const decay = fastExp(CFG.dt / CFG.tausyn), leakK = CFG.dt / CFG.tau;
  // Hot loop : tout en locaux (pas de lookup d'objet/propriété par neurone)
  const VTH = CFG.vth, VREST = CFG.vrest, VRST = CFG.vreset;
  const REFR = CFG.refr, STIMP = CFG.stimP, NM = net.nm, MOTB = n - NM;
  const rnd = Math.random;
  const V = new Float32Array(n * B).fill(VREST);
  const G = new Float32Array(n * B), RF = new Uint8Array(n * B);
  let Sp = new Uint8Array(n * B);
  const delay = [new Uint8Array(n * B), new Uint8Array(n * B)];
  const motorSum = new Float32Array(B * NM), framesAll = [];
  let nRec = 0;
  for (let s = 0; s < steps; s++) {
    const delayed = delay.shift();
    delay.push(Sp.slice());
    for (let b = 0; b < B; b++) {
      const off = b * n;
      // 1) décroissance synaptique (tenseur plein, pas cher)
      for (let i = 0; i < n; i++) G[off + i] *= decay;
      // 2) scatter événementiel : seuls les pré-synaptiques actifs paient.
      // ~6-7 % d'activité mesurée → ~15× moins de MAC que le full CSR.
      for (let pre = 0; pre < n; pre++) {
        if (!delayed[off + pre]) continue;
        for (let k = colptr[pre]; k < colptr[pre + 1]; k++) {
          G[off + rowidx[k]] += cvals[k] * W;
        }
      }
      // 3) intégration + spikes
      for (let post = 0; post < n; post++) {
        const idx = off + post;
        V[idx] += (-(V[idx] - VREST) + G[idx]) * leakK;
        let sp = 0;
        if (RF[idx] > 0) RF[idx]--;
        else if (V[idx] >= VTH) { V[idx] = VRST; RF[idx] = REFR; sp = 1; }
        if (post < ns) {
          const p = currents[b * ns + post] * STIMP;
          if (rnd() < (p > 1 ? 1 : p > 0 ? p : 0)) sp = 1;
        }
        Sp[idx] = sp;
      }
    }
    if (s % recordEvery === 0) {
      nRec++;
      const fr = [];
      for (let b = 0; b < B; b++) {
        const off = b * n, col = [];
        for (let m = 0; m < NM; m++) motorSum[b * NM + m] += Sp[off + MOTB + m];
        if (collectFrames) for (let i = 0; i < n && col.length < 2000; i++) if (Sp[off + i]) col.push(i);
        fr.push(col);
      }
      if (collectFrames) framesAll.push(fr);
    }
  }
  const motorMean = new Float32Array(B * NM);
  for (let k = 0; k < motorMean.length; k++) motorMean[k] = motorSum[k] / Math.max(nRec, 1);
  return { motorMean, framesAll, T: nRec };
}

function buildReadout(nIn, seed) {
  const randn = makeRandn(seed), w = new Float32Array(nIn);
  for (let k = 0; k < nIn; k++) w[k] = randn() / Math.sqrt(nIn);
  return w;
}

function scoreMotor(w, motorMean, B, nm) {
  const out = new Float32Array(B);
  for (let b = 0; b < B; b++) {
    let dot = 0, nrm = 0;
    for (let m = 0; m < nm; m++) { const v = motorMean[b * nm + m]; dot += v * w[m]; nrm += v * v; }
    nrm = Math.sqrt(nrm) || 1;
    out[b] = dot / nrm; // pattern normalisé (readout.py)
  }
  return out;
}

let NET = null, ENC = null, READOUT = null, STIM = null;

function init(n) {
  NET = buildNet(n || CFG.n, 42);
  ENC = buildEncoder(NET.ns, 42);
  READOUT = buildReadout(NET.nm, 1234);
  const randn = makeRandn(7), proj = new Float32Array(8 * NET.ns), s = 1 / Math.sqrt(8);
  for (let k = 0; k < proj.length; k++) proj[k] = randn() * s;
  STIM = { proj };
  return { n: NET.n, ns: NET.ns, nm: NET.nm, nnz: NET.nnz };
}

function stimEncode(action, gain) {
  const r = RECIPES[action] || RECIPES.chess, out = new Float32Array(NET.ns);
  for (let j = 0; j < NET.ns; j++) {
    let v = 0;
    for (let c = 0; c < 8; c++) v += (r[CHANNELS[c]] || 0) * STIM.proj[c * NET.ns + j];
    out[j] = v * (gain || 5);
  }
  return out;
}

function handle(e) {
  const m = e.data;
  if (m.type === 'init') {
    const info = init(m.n);
    return { id: m.id, type: 'ready', ...info };
  }
  if (m.type === 'score') {
    const feats = m.boards.map(b => features(b, CFG.nf));
    const cur = encodeBatch(ENC, feats);
    const r = simulate(NET, cur, m.boards.length, m.steps || 60, m.record || 10, true);
    const scores = scoreMotor(READOUT, r.motorMean, m.boards.length, NET.nm);
    const means = [];
    for (let b = 0; b < m.boards.length; b++) {
      let s = 0;
      for (let mm = 0; mm < NET.nm; mm++) s += r.motorMean[b * NET.nm + mm];
      means.push(s / NET.nm);
    }
    return { id: m.id, type: 'scores', scores: Array.from(scores), motorMeans: means, framesAll: r.framesAll, T: r.T };
  }
  if (m.type === 'react') {
    const cur = stimEncode(m.action);
    // B=1 : on encapsule le vecteur (ns,) en batch de 1
    const r = simulate(NET, cur, 1, m.steps || 100, 10, false);
    const mean = r.motorMean.reduce((a, b) => a + b, 0) / r.motorMean.length;
    return { id: m.id, type: 'reaction', strength: Math.tanh(mean * 300) };
  }
  return { id: m.id, type: 'error', detail: 'inconnu' };
}

if (typeof self !== 'undefined' && typeof importScripts === 'function') {
  self.onmessage = (e) => self.postMessage(handle(e));
}

function calibrate(list) {
  // Balayage Wsyn : cible sensoriels ~10-60 Hz, moteurs 0.5-8 Hz, scores distincts.
  // (miroir de la calibration Shiu : 80 % de la réponse max — ici on règle le
  // régime, pas un réflexe ; le worker n'a pas de MN9)
  init(5000);
  const mk = (mat) => ({ pieces: [[12, 5, 1], [52, 6, 0]], legal: 20, check: 0, mate: 0, stale: 0, turn: 1, fullmove: 1, material: mat });
  const feats = [mk(0), mk(8)].map(f => features(f, CFG.nf));
  const cur = encodeBatch(ENC, feats);
  for (const W of (list || [0.05, 0.164, 0.5])) {
    const r = simulate(NET, cur, 2, 100, 10, true, W);
    let ns = 0, nm = 0, tot = 0;
    for (const fr of r.framesAll) for (const col of fr) {
      tot += col.length;
      for (const i of col) { if (i < NET.ns) ns++; else if (i >= NET.n - NET.nm) nm++; }
    }
    const sec = 100 / 1000 * 10 / r.T; // s simulées enregistrées ≈ 100 ms
    const sc = scoreMotor(READOUT, r.motorMean, 2, NET.nm);
    console.log('wsyn=' + W + ' sensHz=' + (ns / NET.ns / 0.1).toFixed(1) +
      ' motHz=' + (nm / NET.nm / 0.1).toFixed(1) +
      ' scores=' + Array.from(sc).map(s => s.toFixed(3)).join(','));
  }
}
function selftest() {
  init(5000); // plein régime (degré entrant ~25 comme backend ; n petit = silence)
  const mk = (mat) => ({ pieces: [[12, 5, 1], [52, 6, 0]], legal: 20, check: 0, mate: 0, stale: 0, turn: 1, fullmove: 1, material: mat });
  const a = handle({ data: { id: 1, type: 'score', boards: [mk(0), mk(8)], steps: 20, record: 10 } });
  // Déterminisme structurel : même seed → même net et même encodage
  // (la SIMU alpha est poissonienne comme le backend : deux runs diffèrent,
  // c'est la biologie, pas un bug — cf. bruit d'humeur chess_noise)
  const n1 = buildNet(5000, 42).nnz, n2 = buildNet(5000, 42).nnz;
  const e1 = encodeBatch(ENC, [features(mk(0), CFG.nf)]);
  const e2 = encodeBatch(ENC, [features(mk(0), CFG.nf)]);
  const det = n1 === n2 && e1.length === e2.length &&
    e1.every((v, i) => v === e2[i]);
  const diff = Math.abs(a.scores[0] - a.scores[1]) > 1e-9;
  const r = handle({ data: { id: 3, type: 'react', action: 'feed', steps: 30 } });
  console.log('scores:', a.scores.map(s => s.toFixed(4)), '| framesT:', a.T, '| react:', r.strength.toFixed(3));
  if (!det) throw new Error('ENCODAGE NON DÉTERMINISTE');
  if (!diff) throw new Error('positions indistinguables (rank-1 local)');
  if (!(r.strength >= 0 && r.strength <= 1)) throw new Error('strength hors borne');
  console.log('WORKER OK : structure déterministe, discriminant, borné');
}

if (typeof module !== 'undefined') module.exports = { init, handle, selftest, calibrate, features, encodeBatch, buildNet, simulate, scoreMotor, Spear: (typeof Spear !== 'undefined' ? Spear : null) };
