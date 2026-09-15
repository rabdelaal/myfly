// Visualisation du connectome en WebGL pur
// Les positions/catégories réelles viennent de GET /api/connectome quand dispo.
class ConnectomeView {
  constructor(canvas, nNeurons = 5000, positions = null, categories = null, hotspots = null) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', { antialias: false, alpha: false });
    if (!this.gl) throw new Error('WebGL unavailable in this browser');
    this.n = nNeurons;

    // Positions : réelles (normalisées par le backend) ou aléatoires
    const pos = new Float32Array(this.n * 3);
    const colors = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n; i++) {
      if (positions && i < positions.length) {
        pos[i*3+0] = positions[i][0];
        pos[i*3+1] = positions[i][2]; // plan z/y échangé pour l'affichage
        pos[i*3+2] = positions[i][1];
      } else {
        pos[i*3+0] = (Math.random() - 0.5) * 10;
        pos[i*3+1] = (Math.random() - 0.5) * 10;
        pos[i*3+2] = (Math.random() - 0.5) * 10;
      }
      // Couleur par catégorie : 0 interneurone, 1 sensoriel, 2 moteur, 3 descendant
      // Hotspot dimorphe (Cell 2026) : teinte rose — les sous-réseaux denses
      // fru/dsx s'allument en grappes (Fig. 7 du papier).
      const cat = categories ? categories[i] : 0;
      const hot = hotspots && hotspots[i] ? 0.45 : 0.0;
      switch (cat) {
        case 1: colors[i*3+0] = 0.2 + hot*0.6; colors[i*3+1] = 0.7 - hot*0.3; colors[i*3+2] = 1.0; break; // bleu→rose
        case 2: colors[i*3+0] = 0.3 + hot*0.5; colors[i*3+1] = 1.0 - hot*0.4; colors[i*3+2] = 0.5 + hot*0.3; break;
        case 3: colors[i*3+0] = 1.0; colors[i*3+1] = 0.6 - hot*0.3; colors[i*3+2] = 0.2 + hot*0.5; break;
        default: colors[i*3+0] = 0.55 + hot*0.35; colors[i*3+1] = 0.4 - hot*0.15; colors[i*3+2] = 0.9; break;
      }
    }
    this.positions = pos;
    this.colors = colors;
    this.baseColors = colors.slice(); // original, pour la grisation lésion
    this.activation = new Float32Array(this.n); // 0..1

    this._initGL();
    this._animate();
  }

  _initGL() {
    const gl = this.gl;
    this._glErrors = [];

    const vs = `
      attribute vec3 aPos;
      attribute vec3 aColor;
      attribute float aAct;
      uniform mat4 projectionMatrix;
      uniform mat4 modelViewMatrix;
      varying vec3 vColor;
      varying float vAct;
      void main() {
        vColor = aColor;
        vAct = aAct;
        vec4 mv = modelViewMatrix * vec4(aPos, 1.0);
        gl_Position = projectionMatrix * mv;
        float size = 1.6 + aAct * 5.0;
        gl_PointSize = size * (14.0 / -mv.z);
      }
    `;
    const fs = `
      precision mediump float;
      varying vec3 vColor;
      varying float vAct;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        if (d > 0.5) discard;
        float core = smoothstep(0.5, 0.1, d);
        float alpha = mix(0.55, 1.0, vAct) * core;
        vec3 col = vColor + vec3(vAct * 0.7, vAct * 0.4, 0.0);
        gl_FragColor = vec4(col, alpha);
      }
    `;

    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        this._glErrors.push(gl.getShaderInfoLog(s) || 'erreur de compilation shader');
      }
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      this._glErrors.push(gl.getProgramInfoLog(prog) || 'erreur de link programme');
    }
    gl.useProgram(prog);
    this.prog = prog;

    // Buffers
    const mkBuf = (data, name, size) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const loc = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return b;
    };
    this.pointBuffers = {
      pos: mkBuf(this.positions, 'aPos', 3),
      col: mkBuf(this.colors, 'aColor', 3),
      act: mkBuf(this.activation, 'aAct', 1),
    };
    this.pointLocs = {
      pos: gl.getAttribLocation(prog, 'aPos'),
      col: gl.getAttribLocation(prog, 'aColor'),
      act: gl.getAttribLocation(prog, 'aAct'),
    };

    // Synapses d'abord (leurs uniforms sont utilisés par la caméra),
    // puis caméra fixe ou orbite automatique
    this.edgeCount = 0;
    this._initEdges();
    this.autoOrbit = false;
    this.orbitAngle = 0;
    this._setCamera();
  }

  _initEdges() {
    const gl = this.gl;
    const vs = `
      attribute vec3 aPos;
      attribute vec3 aColor;
      attribute float aAlpha;
      uniform mat4 projectionMatrix;
      uniform mat4 modelViewMatrix;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vColor = aColor;
        vAlpha = aAlpha;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(aPos, 1.0);
      }
    `;
    const fs = `
      precision mediump float;
      varying vec3 vColor;
      varying float vAlpha;
      void main() { gl_FragColor = vec4(vColor, vAlpha); }
    `;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    this.edgeProg = prog;
    this.edgeBuffers = {};
    this.edgeUniforms = {
      projectionMatrix: gl.getUniformLocation(prog, 'projectionMatrix'),
      modelViewMatrix: gl.getUniformLocation(prog, 'modelViewMatrix'),
    };
  }

  /**
   * edges : {a: [i...], b: [i...], sign: [0|1...]} — indices dans le sous-
   * ensemble affiché. Les synapses s'allument quand leurs deux neurones
   * sont actifs (excitatrices chaudes, inhibitrices froides).
   */
  setEdges(edges) {
    const gl = this.gl;
    const n = edges.a.length;
    this.edgeCount = n;
    if (!n) return;
    this.edgeA = edges.a;
    this.edgeB = edges.b;
    const pos = new Float32Array(n * 6);
    const col = new Float32Array(n * 6);
    this.edgeAlpha = new Float32Array(n * 2);
    for (let e = 0; e < n; e++) {
      const A = edges.a[e], B = edges.b[e];
      pos.set([this.positions[A*3], this.positions[A*3+1], this.positions[A*3+2],
               this.positions[B*3], this.positions[B*3+1], this.positions[B*3+2]], e * 6);
      const warm = edges.sign[e] === 1;
      const c = warm ? [0.9, 0.45, 0.15] : [0.25, 0.55, 1.0];
      col.set([...c, ...c], e * 6);
    }
    const mk = (data, name, size) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const loc = gl.getAttribLocation(this.edgeProg, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return b;
    };
    this.edgeBuffers.pos = mk(pos, 'aPos', 3);
    this.edgeBuffers.col = mk(col, 'aColor', 3);
    this.edgeBuffers.alpha = mk(this.edgeAlpha, 'aAlpha', 1);
  }

  _drawEdges() {
    if (!this.edgeCount || !this.edgeAlpha) return;
    const gl = this.gl;
    // Luminosité de chaque synapse = activation du neurone le moins actif
    const act = this.activation, A = this.edgeAlpha;
    for (let e = 0; e < this.edgeCount; e++) {
      const a = act[this.edgeA[e]] || 0;
      const b = act[this.edgeB[e]] || 0;
      const glow = Math.min(a, b);
      A[2*e] = 0.05 + 0.75 * glow;
      A[2*e+1] = 0.05 + 0.75 * glow;
    }
    gl.useProgram(this.edgeProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuffers.pos);
    gl.vertexAttribPointer(gl.getAttribLocation(this.edgeProg, 'aPos'), 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(gl.getAttribLocation(this.edgeProg, 'aPos'));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuffers.col);
    gl.vertexAttribPointer(gl.getAttribLocation(this.edgeProg, 'aColor'), 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(gl.getAttribLocation(this.edgeProg, 'aColor'));
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeBuffers.alpha);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, A);
    gl.vertexAttribPointer(gl.getAttribLocation(this.edgeProg, 'aAlpha'), 1, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(gl.getAttribLocation(this.edgeProg, 'aAlpha'));
    gl.uniformMatrix4fv(this.edgeUniforms.projectionMatrix, false,
                        this._lastProjection || this._identity());
    gl.uniformMatrix4fv(this.edgeUniforms.modelViewMatrix, false,
                        this._lastView || this._identity());
    gl.drawArrays(gl.LINES, 0, this.edgeCount * 2);
  }

  _identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }

  _setCamera() {
    this._applyCamera();
  }

  _applyCamera() {
    const gl = this.gl;
    const aspect = (this.canvas.width / this.canvas.height) || 1;
    const proj = this._perspective(60 * Math.PI / 180, aspect, 0.1, 100);
    let view;
    if (this.autoOrbit) {
      const t = this.orbitAngle;
      const eye = [16 * Math.sin(t), 4 * Math.sin(t * 0.6) + 3, 16 * Math.cos(t)];
      view = this._lookAt(eye, [0, 0, 0], [0, 1, 0]);
    } else {
      view = this._lookAt([0, 0, 11.5], [0, 0, 0], [0, 1, 0]);
    }
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'projectionMatrix'), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, 'modelViewMatrix'), false, view);
    if (this.edgeUniforms) {
      gl.useProgram(this.edgeProg);
      gl.uniformMatrix4fv(this.edgeUniforms.projectionMatrix, false, proj);
      gl.uniformMatrix4fv(this.edgeUniforms.modelViewMatrix, false, view);
    }
    this._lastProjection = proj;
    this._lastView = view;
  }

  _perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0,
    ]);
  }

  _lookAt(eye, center, up) {
    const [ex, ey, ez] = eye, [cx, cy, cz] = center;
    let zx = ex - cx, zy = ey - cy, zz = ez - cz;
    const zl = Math.hypot(zx, zy, zz);
    zx /= zl; zy /= zl; zz /= zl;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    const xl = Math.hypot(xx, xy, xz) || 1;
    xx /= xl; xy /= xl; xz /= xl;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx*ex + xy*ey + xz*ez),
      -(yx*ex + yy*ey + yz*ez),
      -(zx*ex + zy*ey + zz*ez), 1,
    ]);
  }

  setActivation(indices) {
    // Décroissance
    for (let i = 0; i < this.n; i++) this.activation[i] *= 0.85;
    // Activer les indices reçus
    for (const idx of indices) {
      if (idx < this.n) this.activation[idx] = 1.0;
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffers.act);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.activation);
  }

  /**
   * ⚗️ Labo : grise les neurones lésés. displayIndices : positions dans le
   * sous-ensemble affiché (ou null pour tout restaurer). Un neurone grisé ne
   * s'allume plus visiblement — côté simulation il ne tire de toute façon plus.
   */
  setLesion(displayIndices) {
    const has = displayIndices && displayIndices.length
      ? new Set(displayIndices) : null;
    const base = this.baseColors;
    for (let i = 0; i < this.n; i++) {
      const dim = has && has.has(i);
      this.colors[i*3+0] = dim ? base[i*3+0] * 0.18 : base[i*3+0];
      this.colors[i*3+1] = dim ? base[i*3+1] * 0.18 : base[i*3+1];
      this.colors[i*3+2] = dim ? base[i*3+2] * 0.18 : base[i*3+2];
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffers.col);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.colors);
  }

  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.02, 0.02, 0.04, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    if (this.autoOrbit) {
      this.orbitAngle += 0.0035;
      this._applyCamera();
    }

    // 1. Synapses (lignes), puis 2. neurones (points additifs)
    this._drawEdges();
    gl.useProgram(this.prog);
    const pb = this.pointBuffers, pl = this.pointLocs;
    gl.bindBuffer(gl.ARRAY_BUFFER, pb.pos);
    gl.vertexAttribPointer(pl.pos, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(pl.pos);
    gl.bindBuffer(gl.ARRAY_BUFFER, pb.col);
    gl.vertexAttribPointer(pl.col, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(pl.col);
    gl.bindBuffer(gl.ARRAY_BUFFER, pb.act);
    gl.vertexAttribPointer(pl.act, 1, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(pl.act);
    gl.drawArrays(gl.POINTS, 0, this.n);
  }

  resize() {
    this.canvas.width = this.canvas.clientWidth * window.devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * window.devicePixelRatio;
    this._setCamera();
  }

  destroy() {
    // Stoppe la boucle rAF (le théâtre en recrée une à chaque ouverture :
    // sans ça, les contextes WebGL s'empilent en arrière-plan).
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    const gl = this.gl;
    if (gl && gl.getExtension('WEBGL_lose_context')) {
      gl.getExtension('WEBGL_lose_context').loseContext();
    }
  }
}

window.ConnectomeView = ConnectomeView;
