// Spear math — ports JS des formules découvertes par régression symbolique,
// VÉRIFIÉES numériquement (maxerr/R² en commentaire). Zéro transcendantale
// sauf mention : c'est là le gain (pas de libm scalaire dans le hot path).
// Les exports WASM/MISRA-C d'origine existent côté moteur ; ici les mêmes
// formules en JS pur pour le mode 100 % navigateur (candidat naturel au
// port WASM : fonctions pures, sans alloc, branchless).
const Spear = {
  // exp(-x) sur [0,1] : 4.2× vs Math.exp, R² 0.970, maxerr 0.067.
  // Usage : facteur de leak exact 1-fastExp(dt/tau) dans le LIF (worker).
  // Hors [0,1] : DÉGRADÉ (linéaire) — ne pas utiliser tel quel.
  fastExp(x) {
    const t = Math.min(0.89011, 0.174912 + x);
    return Math.sqrt(Math.abs(-1.078939 + t));
  },
  // Fog three.js exp2 sur [0,2] : 5.5×, R² 0.94, maxerr 0.137.
  // OK brouillard lointain/LOD, PAS pour leplan héro.
  fogFactor(x) {
    return 0.925481 - Math.abs(0.492122 * Math.abs(x));
  },
  // Segment gamma sRGB x^0.4167 sur (0,1] : 5.13×, R² 0.987, maxerr 0.039.
  // Le quasi-gratuit du lot (pipeline couleur du connectome).
  srgbEncodeSeg(x) {
    const s1 = Math.sqrt(Math.abs(x));
    const s2 = Math.sqrt(Math.abs(s1));
    const c = s2 * s2 * s2;
    return Math.sqrt(Math.abs(c));
  },
  // Fresnel-Schlick F0=0.04 sur [0,1] : 14.67×, R² 0.90, maxerr 0.147.
  // Biais aux angles rasants — périphérie uniquement.
  fresnel(x) {
    return Math.max(0.0, 0.901819 - x - x);
  },
  // Back ease-out (p5.js) sur [0,1] : 1.5×, R² 0.984 — animations UI.
  backEaseOut(x) {
    return Math.min(1.092633 * x + Math.sqrt(Math.abs(-x)), 1.055068);
  },
  // ACES Narkowicz sur [0,2] : 1.83×, R² 0.99 — tonemap canvas.
  aces(x) {
    return Math.min(Math.sqrt(Math.abs(Math.min(0.773522, x * 0.672023))), x * 1.455366);
  },
};
if (typeof window !== 'undefined') window.Spear = Spear;
