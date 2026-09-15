// Plateau d'échecs 3D avec Three.js
// - Rendu des pièces depuis le FEN
// - Sélection / déplacement des pièces par raycasting (clic = clic sans glisser)
// - Surbrillance des cases légales

const FILES = 'abcdefgh';

class Chess3D {
  constructor(canvas, onSquareClick = null) {
    this.canvas = canvas;
    this.onSquareClick = onSquareClick;
    this.selected = null; // index de case sélectionnée (0..63)

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x06060a);

    this.camera = new THREE.PerspectiveCamera(
      45, canvas.clientWidth / canvas.clientHeight, 0.1, 1000
    );
    this.camera.position.set(8, 10, 8);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // Lumière
    const amb = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(amb);
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(5, 10, 5);
    this.scene.add(dir);

    this._buildBoard();
    this._buildPieces();
    this._buildHighlights();

    // Contrôle orbite simple + picking
    this._setupOrbit();
    this._animate();
  }

  // --- Correspondance case (0..63, a1=0) ↔ position 3D ---
  squareToPos(sq) {
    const file = sq % 8, rank = Math.floor(sq / 8);
    return { x: file - 3.5, z: (7 - rank) - 3.5 };
  }

  posToSquare(x, z) {
    const file = Math.round(x + 3.5);
    const rank = 7 - Math.round(z + 3.5);
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
    return rank * 8 + file;
  }

  _buildBoard() {
    const size = 1;
    this.tiles = [];
    const tileGeo = new THREE.BoxGeometry(size, 0.1, size);
    for (let sq = 0; sq < 64; sq++) {
      const { x, z } = this.squareToPos(sq);
      const file = sq % 8, rank = Math.floor(sq / 8);
      const color = (file + rank) % 2 === 0 ? 0x3a3a4a : 0x1a1a24;
      const mat = new THREE.MeshStandardMaterial({ color });
      const tile = new THREE.Mesh(tileGeo, mat);
      tile.position.set(x, 0, z);
      tile.userData.square = sq;
      this.scene.add(tile);
      this.tiles.push(tile);
    }
  }

  _buildPieces() {
    this.pieces = new THREE.Group();
    this.scene.add(this.pieces);
  }

  _buildHighlights() {
    // Disques semi-transparents posés sur les cases
    this.highlightGroup = new THREE.Group();
    this.scene.add(this.highlightGroup);
    const geo = new THREE.CircleGeometry(0.35, 24);
    geo.rotateX(-Math.PI / 2);
    this.selectMat = new THREE.MeshBasicMaterial({ color: 0x6affb0, transparent: true, opacity: 0.5 });
    this.targetMat = new THREE.MeshBasicMaterial({ color: 0x4a6eff, transparent: true, opacity: 0.45 });
    this.captureMat = new THREE.MeshBasicMaterial({ color: 0xff5a5a, transparent: true, opacity: 0.5 });
  }

  clearHighlights() {
    while (this.highlightGroup.children.length) {
      this.highlightGroup.remove(this.highlightGroup.children[0]);
    }
  }

  showSelection(sq) {
    this.clearHighlights();
    this.selected = sq;
    const { x, z } = this.squareToPos(sq);
    const m = new THREE.Mesh(new THREE.CircleGeometry(0.42, 24).rotateX(-Math.PI / 2), this.selectMat);
    m.position.set(x, 0.06, z);
    this.highlightGroup.add(m);
  }

  showTargets(moves) {
    // moves : [{ to: sq, capture: bool }]
    for (const { to, capture } of moves) {
      const { x, z } = this.squareToPos(to);
      const mat = capture ? this.captureMat : this.targetMat;
      const geo = capture
        ? new THREE.RingGeometry(0.3, 0.45, 24).rotateX(-Math.PI / 2)
        : new THREE.CircleGeometry(0.25, 24).rotateX(-Math.PI / 2);
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x, 0.06, z);
      this.highlightGroup.add(m);
    }
  }

  setPosition(fen) {
    // Nettoyer
    while (this.pieces.children.length) {
      this.pieces.remove(this.pieces.children[0]);
    }
    // Parser le FEN (simplifié, ne gère que la partie pièces)
    const rows = fen.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
      let col = 0;
      for (const ch of rows[r]) {
        if (/\d/.test(ch)) { col += parseInt(ch); continue; }
        const isWhite = ch === ch.toUpperCase();
        const type = ch.toLowerCase();
        const rank = 7 - r; // rangée FEN 0 = rang 8
        const sq = rank * 8 + col;
        const mesh = this._makePiece(type, isWhite);
        mesh.position.set(col - 3.5, 0.4, r - 3.5);
        mesh.userData.square = sq;
        mesh.userData.type = type;
        mesh.userData.isWhite = isWhite;
        this.pieces.add(mesh);
        col++;
      }
    }
    this.clearHighlights();
    this.selected = null;
  }

  _makePiece(type, isWhite) {
    const color = isWhite ? 0xf0f0f0 : 0x4a4a66;
    const mat = new THREE.MeshStandardMaterial({
      color, roughness: 0.5,
      emissive: isWhite ? 0x111111 : 0x1a1a2e,
    });
    let geo;
    switch (type) {
      case 'p': geo = new THREE.ConeGeometry(0.25, 0.6, 12); break;
      case 'r': geo = new THREE.BoxGeometry(0.5, 0.6, 0.5); break;
      case 'n': geo = new THREE.ConeGeometry(0.3, 0.7, 4); break;
      case 'b': geo = new THREE.ConeGeometry(0.3, 0.7, 8); break;
      case 'q': geo = new THREE.CylinderGeometry(0.35, 0.35, 0.7, 12); break;
      case 'k': geo = new THREE.CylinderGeometry(0.4, 0.4, 0.8, 12); break;
      default:  geo = new THREE.SphereGeometry(0.3, 12, 12);
    }
    return new THREE.Mesh(geo, mat);
  }

  // --- Picking : clic (sans glisser) sur une case ou une pièce ---
  _pickSquare(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    const hits = raycaster.intersectObjects([...this.pieces.children, ...this.tiles], false);
    for (const hit of hits) {
      const sq = hit.object.userData.square;
      if (sq !== undefined) return sq;
    }
    return null;
  }

  _setupOrbit() {
    // Pointer Events unifiés : souris + tactile. Tap (<6px, <400ms) = clic
    // case ; glisser = orbite. touch-action:none (CSS) bloque le scroll.
    let isDown = false, px = 0, py = 0, downX = 0, downY = 0, downT = 0;
    let theta = Math.PI / 4, phi = Math.PI / 3, radius = 11.5;
    const update = () => {
      this.camera.position.x = radius * Math.sin(phi) * Math.cos(theta);
      this.camera.position.y = radius * Math.cos(phi);
      this.camera.position.z = radius * Math.sin(phi) * Math.sin(theta);
      this.camera.lookAt(0, 0, 0);
    };
    update();

    this.canvas.addEventListener('pointerdown', (e) => {
      isDown = true;
      px = downX = e.clientX; py = downY = e.clientY;
      downT = performance.now();
      try { this.canvas.setPointerCapture(e.pointerId); } catch (err) { /* noop */ }
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (!isDown) return;
      const dx = e.clientX - px, dy = e.clientY - py;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) {
        theta += dx * 0.01;
        phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, phi - dy * 0.01));
        update();
      }
      px = e.clientX; py = e.clientY;
    });
    const up = (e) => {
      if (!isDown) return;
      isDown = false;
      const dist = Math.hypot(e.clientX - downX, e.clientY - downY);
      if (dist < 6 && performance.now() - downT < 600 && this.onSquareClick) {
        this.onSquareClick(this._pickSquare(e.clientX, e.clientY));
      }
    };
    this.canvas.addEventListener('pointerup', up);
    this.canvas.addEventListener('pointercancel', () => { isDown = false; });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      radius = Math.max(6, Math.min(30, radius + e.deltaY * 0.01));
      update();
    }, { passive: false });
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    // Dimensionnement explicite : en % pur, le canvas restait à 0px quand
    // l'onglet était masqué à l'init (clientWidth = 0). On mesure le parent.
    const parent = this.canvas.parentElement;
    const w = parent ? parent.clientWidth : this.canvas.clientWidth;
    const h = this.canvas.clientHeight || (parent ? parent.clientHeight : 0);
    if (w) this.canvas.style.width = w + 'px';
    if (h) this.canvas.style.height = h + 'px';
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if (!cw || !ch) return;
    this.renderer.setSize(cw, ch);
    this.camera.aspect = cw / ch;
    this.camera.updateProjectionMatrix();
  }
}

window.Chess3D = Chess3D;
