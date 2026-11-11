// The viewport in plain Three.js (PLAN.md, Phase 3).
//
// Node coordinates, displacements and nodal values live in shared
// BufferAttribute objects used by surfaces, edges, beams and picking.
// A new field or a new frame only rewrites the arrays; no new
// geometries are created per frame. The image is only redrawn when something has changed.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MeshData, Symbols } from "../engine/Engine";
import { colormap, type ColormapName } from "./colormaps";

export type ViewDir = "iso" | "front" | "back" | "top" | "bottom" | "left" | "right";

export interface Hit {
  kind: "element" | "node";
  index: number;
}

export interface ViewOptions {
  surfaces: boolean;
  edges: boolean;
  /** Beams with their cross-section and shells with their thickness, in 3D. */
  profiles: boolean;
  undeformed: boolean;
  symbols: boolean;
  levels: number;
  colormap: ColormapName;
  pickMode: "element" | "node";
  dark: boolean;
}

/** Value that means "field missing" in the shader. */
const NONE = -1e30;

const surfaceVert = /* glsl */ `
in vec3 disp;
in float val;
uniform float scale;
out float vVal;
out vec3 vView;
void main() {
  vec3 p = position + scale * disp;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vView = mv.xyz;
  vVal = val;
  gl_Position = projectionMatrix * mv;
}`;

const surfaceFrag = /* glsl */ `
precision highp float;
uniform sampler2D cmap;
uniform float vmin;
uniform float vmax;
uniform float levels;
uniform float useField;
uniform vec3 baseColor;
uniform float opacity;
in float vVal;
in vec3 vView;
out vec4 fragColor;
void main() {
  vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
  float l = 0.42 + 0.58 * abs(n.z);
  vec3 c = baseColor;
  if (useField > 0.5) {
    if (vVal < -1e29) {
      c = vec3(0.5);
    } else {
      float t = clamp((vVal - vmin) / max(vmax - vmin, 1e-30), 0.0, 1.0);
      if (vmax <= vmin) t = 0.5;
      if (levels > 0.5) t = (min(floor(t * levels), levels - 1.0) + 0.5) / levels;
      c = texture(cmap, vec2(t, 0.5)).rgb;
    }
  }
  fragColor = vec4(c * l, opacity);
}`;

const lineVert = /* glsl */ `
in vec3 disp;
uniform float scale;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position + scale * disp, 1.0);
}`;

const lineFrag = /* glsl */ `
precision highp float;
uniform vec3 color;
uniform float opacity;
out vec4 fragColor;
void main() { fragColor = vec4(color, opacity); }`;

// Wide lines (beams, trusses): each segment is a rectangle in the screen plane.
const fatVert = /* glsl */ `
in vec3 pa;
in vec3 pb;
in vec3 da;
in vec3 db;
in float va;
in float vb;
in vec2 corner;
in float pid;
uniform float scale;
uniform float width;
uniform vec2 resolution;
out float vVal;
out float vId;
void main() {
  vec4 a = projectionMatrix * modelViewMatrix * vec4(pa + scale * da, 1.0);
  vec4 b = projectionMatrix * modelViewMatrix * vec4(pb + scale * db, 1.0);
  vec2 sa = a.xy / a.w * resolution * 0.5;
  vec2 sb = b.xy / b.w * resolution * 0.5;
  vec2 d = sb - sa;
  float len = length(d);
  vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 p = corner.x < 0.5 ? a : b;
  vec2 off = (nrm * corner.y + dir * (corner.x < 0.5 ? -0.5 : 0.5)) * width;
  p.xy += off / (resolution * 0.5) * p.w;
  gl_Position = p;
  vVal = corner.x < 0.5 ? va : vb;
  vId = pid;
}`;

const fatFrag = /* glsl */ `
precision highp float;
uniform sampler2D cmap;
uniform float vmin;
uniform float vmax;
uniform float levels;
uniform float useField;
uniform vec3 baseColor;
in float vVal;
in float vId;
out vec4 fragColor;
void main() {
  vec3 c = baseColor;
  if (useField > 0.5 && vVal > -1e29) {
    float t = clamp((vVal - vmin) / max(vmax - vmin, 1e-30), 0.0, 1.0);
    if (vmax <= vmin) t = 0.5;
    if (levels > 0.5) t = (min(floor(t * levels), levels - 1.0) + 0.5) / levels;
    c = texture(cmap, vec2(t, 0.5)).rgb;
  }
  fragColor = vec4(c, 1.0);
}`;

const pickVert = /* glsl */ `
in vec3 disp;
in float pid;
uniform float scale;
out float vId;
void main() {
  vId = pid;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position + scale * disp, 1.0);
}`;

const pickFrag = /* glsl */ `
precision highp float;
in float vId;
out vec4 fragColor;
void main() {
  float id = vId + 0.5;
  float r = mod(id, 256.0);
  float g = mod(floor(id / 256.0), 256.0);
  float b = floor(id / 65536.0);
  fragColor = vec4(floor(r), g, b, 255.0) / 255.0;
}`;

const fatPickFrag = /* glsl */ `
precision highp float;
in float vVal;
in float vId;
out vec4 fragColor;
void main() {
  float id = vId + 0.5;
  fragColor = vec4(floor(mod(id, 256.0)), mod(floor(id / 256.0), 256.0), floor(id / 65536.0), 255.0) / 255.0;
}`;

const DIRS: Record<ViewDir, [THREE.Vector3, THREE.Vector3]> = {
  iso: [new THREE.Vector3(1, 0.8, 1.2), new THREE.Vector3(0, 1, 0)],
  front: [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0)],
  back: [new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1, 0)],
  top: [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, -1)],
  bottom: [new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1)],
  left: [new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0)],
  right: [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0)],
};

/**
 * Loads grouped for labels: nodes with the same direction that lie close to
 * each other (a distributed load, an edge) become one group with the sum. The distance limit
 * is 2.5 × the median distance to the nearest node, at most a quarter of the model's radius.
 */
function loadGroups(items: { n: number; d: THREE.Vector3; mag: number }[], P: Float32Array, radius: number) {
  const out: { center: THREE.Vector3; d: THREE.Vector3; total: number; count: number }[] = [];
  if (items.length > 3000) return out;
  const byDir = new Map<string, typeof items>();
  for (const it of items) {
    const k = [it.d.x, it.d.y, it.d.z].map((x) => x.toFixed(3)).join(",");
    byDir.set(k, [...(byDir.get(k) ?? []), it]);
  }
  const pos = (it: { n: number }) => new THREE.Vector3().fromArray(P, it.n * 3);
  for (const list of byDir.values()) {
    const pts = list.map(pos);
    const nn = pts.map((a, i) => Math.min(...pts.map((b, j) => (i === j ? Infinity : a.distanceTo(b)))));
    const med = [...nn].sort((a, b) => a - b)[Math.floor(nn.length / 2)];
    const tol = Math.min(Number.isFinite(med) ? med * 2.5 : 0, radius * 0.25);
    const group = new Array<number>(list.length).fill(-1);
    let ng = 0;
    for (let i = 0; i < list.length; i++) {
      if (group[i] >= 0) continue;
      group[i] = ng;
      const stack = [i];
      while (stack.length) {
        const a = stack.pop()!;
        for (let j = 0; j < list.length; j++) {
          if (group[j] < 0 && pts[a].distanceTo(pts[j]) <= tol) {
            group[j] = ng;
            stack.push(j);
          }
        }
      }
      ng++;
    }
    for (let g = 0; g < ng; g++) {
      const idx = list.map((_, i) => i).filter((i) => group[i] === g);
      const center = new THREE.Vector3();
      idx.forEach((i) => center.add(pts[i]));
      center.divideScalar(idx.length);
      out.push({ center, d: list[idx[0]].d, total: idx.reduce((s, i) => s + list[i].mag, 0), count: idx.length });
    }
  }
  return out;
}

export class Viewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private pickScene = new THREE.Scene();
  private camera: THREE.OrthographicCamera;
  private controls: OrbitControls;
  private triad: { scene: THREE.Scene; camera: THREE.OrthographicCamera };
  private root = new THREE.Group();
  private cmapTex: THREE.DataTexture;
  private surfMat: THREE.ShaderMaterial;
  private edgeMat: THREE.ShaderMaterial;
  private ghostMat: THREE.ShaderMaterial;
  private fatMat: THREE.ShaderMaterial;
  private pointMat: THREE.ShaderMaterial;
  private hiMat: THREE.ShaderMaterial;
  private pickMat: THREE.ShaderMaterial;
  private fatPickMat: THREE.ShaderMaterial;
  private mesh: MeshData | null = null;
  private pos: THREE.BufferAttribute | null = null;
  private disp: THREE.BufferAttribute | null = null;
  private val: THREE.BufferAttribute | null = null;
  private fat: THREE.Mesh | null = null;
  /** Extruded profiles and thick shells: one vertex per corner, with the node it follows. */
  private solid: { mesh: THREE.Mesh; pick: THREE.Mesh; node: Uint32Array } | null = null;
  private solidMat: THREE.ShaderMaterial;
  private symbolGroup = new THREE.Group();
  private highlight = new THREE.Group();
  /** Nodes hit by a selection in the tool panel (preview). */
  private marks = new THREE.Group();
  private pickTarget = new THREE.WebGLRenderTarget(1, 1);
  private center = new THREE.Vector3();
  private box = new THREE.Box3();
  private radius = 1;
  private used = new Uint8Array(0);
  private autoFit = true;
  private is2d = false;
  private frameReq = 0;
  private animating: { kind: "mode"; base: number; t0: number } | null = null;
  private opts: ViewOptions;
  private ro: ResizeObserver;
  private down: { x: number; y: number; t: number; touch: boolean } | null = null;
  scale = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private onPick: (h: Hit | null) => void,
    opts: ViewOptions,
  ) {
    this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.autoClear = false;
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.zoomToCursor = true;
    this.controls.rotateSpeed = 0.9;
    this.controls.addEventListener("change", () => this.requestRender());
    // once the user has moved the camera, a resize keeps their view instead of refitting
    this.controls.addEventListener("start", () => (this.autoFit = false));
    this.cmapTex = new THREE.DataTexture(colormap(opts.colormap), 256, 1, THREE.RGBAFormat);
    this.cmapTex.magFilter = THREE.NearestFilter;
    this.cmapTex.minFilter = THREE.NearestFilter;
    this.cmapTex.needsUpdate = true;
    const field = {
      cmap: { value: this.cmapTex },
      vmin: { value: 0 },
      vmax: { value: 1 },
      levels: { value: opts.levels },
      useField: { value: 0 },
      scale: { value: 0 },
    };
    this.surfMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: surfaceVert,
      fragmentShader: surfaceFrag,
      uniforms: { ...field, baseColor: { value: new THREE.Color(0.62, 0.7, 0.8) }, opacity: { value: 1 } },
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    // the same uniforms as the surfaces, but without the polygon offset, so that a shell face
    // on the nodes (OFFSET=SPOS/SNEG) is drawn in front of the plain surface
    this.solidMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: surfaceVert,
      fragmentShader: surfaceFrag,
      uniforms: this.surfMat.uniforms,
      side: THREE.DoubleSide,
    });
    this.edgeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: lineVert,
      fragmentShader: lineFrag,
      uniforms: { scale: field.scale, color: { value: new THREE.Color(0x0b0d12) }, opacity: { value: 0.55 } },
      transparent: true,
      depthWrite: false,
    });
    this.ghostMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: lineVert,
      fragmentShader: lineFrag,
      uniforms: { scale: { value: 0 }, color: { value: new THREE.Color(0x8892a6) }, opacity: { value: 0.35 } },
      transparent: true,
      depthWrite: false,
    });
    const res = { value: new THREE.Vector2(1, 1) };
    this.fatMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fatVert,
      fragmentShader: fatFrag,
      uniforms: { ...field, baseColor: { value: new THREE.Color(0.85, 0.88, 0.95) }, width: { value: 3.5 }, resolution: res },
      side: THREE.DoubleSide,
    });
    this.pointMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: lineVert.replace("gl_Position =", "gl_PointSize = 9.0;\n  gl_Position ="),
      fragmentShader: lineFrag,
      uniforms: { scale: field.scale, color: { value: new THREE.Color(0xf5b942) }, opacity: { value: 1 } },
    });
    this.hiMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: lineVert,
      fragmentShader: lineFrag,
      uniforms: { scale: field.scale, color: { value: new THREE.Color(0xffd84d) }, opacity: { value: 0.55 } },
      transparent: true,
      depthTest: false,
      side: THREE.DoubleSide,
    });
    this.pickMat = new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: pickVert, fragmentShader: pickFrag, uniforms: { scale: field.scale }, side: THREE.DoubleSide });
    this.fatPickMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fatVert,
      fragmentShader: fatPickFrag,
      uniforms: { scale: field.scale, width: { value: 12 }, resolution: res },
      side: THREE.DoubleSide,
    });
    this.scene.add(this.root, this.symbolGroup, this.highlight, this.marks);
    this.triad = this.makeTriad();
    this.applyTheme();
    // the viewport changes size when the console or a side panel opens: refit until the user moves the camera
    this.ro = new ResizeObserver(() => (this.autoFit ? this.fit() : this.resize()));
    this.ro.observe(canvas.parentElement ?? canvas);
    canvas.addEventListener("pointerdown", this.onDown);
    canvas.addEventListener("pointerup", this.onUp);
    this.resize();
  }

  dispose() {
    cancelAnimationFrame(this.frameReq);
    this.ro.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onDown);
    this.canvas.removeEventListener("pointerup", this.onUp);
    this.controls.dispose();
    this.clear();
    this.renderer.dispose();
  }

  private makeTriad() {
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, -10, 10);
    const axes: [THREE.Vector3, number][] = [
      [new THREE.Vector3(1, 0, 0), 0xe5534b],
      [new THREE.Vector3(0, 1, 0), 0x57ab5a],
      [new THREE.Vector3(0, 0, 1), 0x539bf5],
    ];
    axes.forEach(([d, c], i) => {
      scene.add(new THREE.ArrowHelper(d, new THREE.Vector3(), 1.1, c, 0.32, 0.18));
      // axis name past the arrow tip, in the arrow's colour
      const px = 64;
      const cv = document.createElement("canvas");
      cv.width = cv.height = px;
      const ctx = cv.getContext("2d")!;
      ctx.font = `700 ${px * 0.8}px system-ui, "Segoe UI", sans-serif`;
      ctx.fillStyle = `#${c.toString(16).padStart(6, "0")}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("xyz"[i], px / 2, px / 2 + 2);
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
      sp.scale.setScalar(0.5);
      sp.position.copy(d).multiplyScalar(1.32);
      scene.add(sp);
    });
    return { scene, camera };
  }

  private applyTheme() {
    this.renderer.setClearColor(this.opts.dark ? 0x12151b : 0xf2f4f7, 1);
    (this.edgeMat.uniforms.color.value as THREE.Color).set(this.opts.dark ? 0x05070a : 0x1d2330);
    (this.fatMat.uniforms.baseColor.value as THREE.Color).set(this.opts.dark ? 0xd5dbe8 : 0x2f3a4d);
    (this.surfMat.uniforms.baseColor.value as THREE.Color).set(this.opts.dark ? 0x8fa3bf : 0xa9b8cc);
  }

  setOptions(o: Partial<ViewOptions>) {
    const old = this.opts;
    this.opts = { ...this.opts, ...o };
    if (o.colormap && o.colormap !== old.colormap) {
      this.cmapTex.image.data = colormap(o.colormap);
      this.cmapTex.needsUpdate = true;
    }
    this.surfMat.uniforms.levels.value = this.opts.levels;
    if (o.dark !== undefined) this.applyTheme();
    this.updateVisibility();
    this.requestRender();
  }

  private updateVisibility() {
    for (const c of this.root.children) {
      if (c.name === "surface") c.visible = this.opts.surfaces;
      if (c.name === "edges") c.visible = this.opts.edges;
      if (c.name === "profiles") c.visible = this.opts.profiles;
      if (c.name === "ghost") c.visible = this.opts.undeformed && this.scale !== 0;
    }
    this.symbolGroup.visible = this.opts.symbols;
    if (this.solid) this.solid.pick.visible = this.opts.profiles;
  }

  private clear() {
    for (const g of [this.root, this.pickScene, this.symbolGroup, this.highlight, this.marks]) {
      for (const c of [...g.children]) {
        g.remove(c);
        c.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.geometry) m.geometry.dispose();
        });
      }
    }
    this.fat = null;
    this.solid = null;
  }

  /** New geometry. `keepView` keeps the camera (after an edit). */
  setMesh(m: MeshData, keepView = false) {
    this.clear();
    this.mesh = m;
    const n = m.nodes;
    this.pos = new THREE.BufferAttribute(m.positions, 3);
    this.disp = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
    this.val = new THREE.BufferAttribute(new Float32Array(n).fill(NONE), 1);
    this.disp.setUsage(THREE.DynamicDrawUsage);
    this.val.setUsage(THREE.DynamicDrawUsage);
    const shared = () => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", this.pos!);
      g.setAttribute("disp", this.disp!);
      g.setAttribute("val", this.val!);
      return g;
    };
    if (m.tris.length) {
      const g = shared();
      g.setIndex(new THREE.BufferAttribute(m.tris, 1));
      const s = new THREE.Mesh(g, this.surfMat);
      s.name = "surface";
      s.frustumCulled = false;
      this.root.add(s);
    }
    if (m.edges.length) {
      const g = shared();
      g.setIndex(new THREE.BufferAttribute(m.edges, 1));
      const e = new THREE.LineSegments(g, this.edgeMat);
      e.name = "edges";
      e.frustumCulled = false;
      e.renderOrder = 1;
      this.root.add(e);
      const gh = new THREE.LineSegments(g, this.ghostMat);
      gh.name = "ghost";
      gh.frustumCulled = false;
      this.root.add(gh);
    }
    if (m.lines.length) {
      this.fat = this.buildFat(m.lines, m.lineElem);
      this.fat.name = "lines";
      this.root.add(this.fat);
    }
    this.solid = this.buildSolid(m);
    if (this.solid) this.root.add(this.solid.mesh);
    if (m.points.length) {
      const g = shared();
      g.setIndex(new THREE.BufferAttribute(m.points, 1));
      const p = new THREE.Points(g, this.pointMat);
      p.frustumCulled = false;
      this.root.add(p);
    }
    this.buildPick();
    // size and center (used nodes only)
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    const used = new Uint8Array(n);
    for (const arr of [m.tris, m.lines, m.points]) for (const i of arr) used[i] = 1;
    let any = false;
    let z = 0;
    for (let i = 0; i < n; i++) {
      if (!used[i]) continue;
      v.fromArray(m.positions, i * 3);
      box.expandByPoint(v);
      z = Math.max(z, Math.abs(v.z));
      any = true;
    }
    if (!any) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    this.box.copy(box);
    this.used = used;
    box.getCenter(this.center);
    this.radius = Math.max(box.getSize(v).length() / 2, 1e-9);
    // a planar model with 3D elements (a frame or a plate in a plane) is still shown in 3D
    this.is2d = z === 0 && box.max.z === box.min.z && !m.profiles?.threeD;
    this.updateVisibility();
    if (!keepView) {
      this.set2D(this.is2d);
      this.view(this.is2d ? "front" : "iso");
    }
    this.requestRender();
  }

  get bounds() {
    return { center: this.center.clone(), radius: this.radius, is2d: this.is2d };
  }

  private buildFat(lines: Uint32Array, elem: Uint32Array): THREE.Mesh {
    const ns = elem.length;
    const g = new THREE.BufferGeometry();
    const mk = (k: number) => new THREE.BufferAttribute(new Float32Array(ns * 4 * k), k);
    const pa = mk(3),
      pb = mk(3),
      da = mk(3),
      db = mk(3),
      va = mk(1),
      vb = mk(1),
      corner = mk(2),
      pid = mk(1);
    const P = this.mesh!.positions;
    const cs = [
      [0, -1],
      [0, 1],
      [1, -1],
      [1, 1],
    ];
    const idx = new Uint32Array(ns * 6);
    for (let s = 0; s < ns; s++) {
      const a = lines[2 * s],
        b = lines[2 * s + 1];
      for (let k = 0; k < 4; k++) {
        const v = s * 4 + k;
        for (let c = 0; c < 3; c++) {
          pa.array[v * 3 + c] = P[a * 3 + c];
          pb.array[v * 3 + c] = P[b * 3 + c];
        }
        corner.array[v * 2] = cs[k][0];
        corner.array[v * 2 + 1] = cs[k][1];
        va.array[v] = NONE;
        vb.array[v] = NONE;
        pid.array[v] = elem[s] + 1;
      }
      idx.set([s * 4, s * 4 + 2, s * 4 + 1, s * 4 + 1, s * 4 + 2, s * 4 + 3], s * 6);
    }
    for (const [k, a] of Object.entries({ pa, pb, da, db, va, vb, corner, pid })) g.setAttribute(k, a);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    const mesh = new THREE.Mesh(g, this.fatMat);
    mesh.frustumCulled = false;
    return mesh;
  }

  /**
   * Beams extruded with their profile along each line segment (the section's
   * n1/n2 axes in the undeformed position) and shells as slabs with their thickness
   * (sides on the free edges). Each vertex follows the displacement and value of its node.
   */
  private buildSolid(m: MeshData): { mesh: THREE.Mesh; pick: THREE.Mesh; node: Uint32Array } | null {
    const pd = m.profiles;
    if (!pd) return null;
    const P = m.positions;
    const pos: number[] = [];
    const node: number[] = [];
    const ids: number[] = [];
    const idx: number[] = [];
    const LIMIT = 3_000_000;
    const vert = (x: number, y: number, z: number, n: number, e: number) => {
      pos.push(x, y, z);
      node.push(n);
      ids.push(e + 1);
      return node.length - 1;
    };
    const nseg = Math.min(pd.segProfile.length, m.lineElem.length);
    for (let s = 0; s < nseg && node.length < LIMIT; s++) {
      const p = pd.segProfile[s];
      if (p < 0 || !pd.profiles[p]) continue;
      const a = m.lines[2 * s],
        b = m.lines[2 * s + 1],
        e = m.lineElem[s];
      const A = new THREE.Vector3().fromArray(P, a * 3);
      const B = new THREE.Vector3().fromArray(P, b * 3);
      const t = B.clone().sub(A);
      if (t.lengthSq() === 0) continue;
      t.normalize();
      const n1 = new THREE.Vector3().fromArray(pd.segN1, s * 3);
      const n2 = new THREE.Vector3().crossVectors(t, n1);
      for (const poly of pd.profiles[p]) {
        const k = poly.length / 2;
        const ia: number[] = [];
        const ib: number[] = [];
        for (let i = 0; i < k; i++) {
          const ox = n1.x * poly[2 * i] + n2.x * poly[2 * i + 1];
          const oy = n1.y * poly[2 * i] + n2.y * poly[2 * i + 1];
          const oz = n1.z * poly[2 * i] + n2.z * poly[2 * i + 1];
          ia.push(vert(A.x + ox, A.y + oy, A.z + oz, a, e));
          ib.push(vert(B.x + ox, B.y + oy, B.z + oz, b, e));
        }
        for (let i = 1; i < k - 1; i++) idx.push(ia[0], ia[i], ia[i + 1], ib[0], ib[i + 1], ib[i]);
        for (let i = 0; i < k; i++) {
          const j = (i + 1) % k;
          idx.push(ia[i], ia[j], ib[j], ia[i], ib[j], ib[i]);
        }
      }
    }
    // shells: count the edges of the shell triangles; an edge used once is free
    const sh = pd.shell;
    const thick = (e: number) => (e * 5 < sh.length ? sh[e * 5] : 0);
    const key = (x: number, y: number) => (x < y ? `${x},${y}` : `${y},${x}`);
    const count = new Map<string, number>();
    const nt = m.triElem.length;
    for (let t = 0; t < nt; t++) {
      if (!(thick(m.triElem[t]) > 0)) continue;
      for (let k = 0; k < 3; k++) {
        const kk = key(m.tris[t * 3 + k], m.tris[t * 3 + ((k + 1) % 3)]);
        count.set(kk, (count.get(kk) ?? 0) + 1);
      }
    }
    for (let t = 0; t < nt && node.length < LIMIT; t++) {
      const e = m.triElem[t];
      const th = thick(e);
      if (!(th > 0)) continue;
      const z = sh[e * 5 + 1];
      const nx = sh[e * 5 + 2],
        ny = sh[e * 5 + 3],
        nz = sh[e * 5 + 4];
      const top: number[] = [];
      const bot: number[] = [];
      for (let k = 0; k < 3; k++) {
        const n = m.tris[t * 3 + k];
        const [x, y, w] = [P[n * 3], P[n * 3 + 1], P[n * 3 + 2]];
        const zt = z + th / 2,
          zb = z - th / 2;
        top.push(vert(x + nx * zt, y + ny * zt, w + nz * zt, n, e));
        bot.push(vert(x + nx * zb, y + ny * zb, w + nz * zb, n, e));
      }
      idx.push(top[0], top[1], top[2], bot[0], bot[2], bot[1]);
      for (let k = 0; k < 3; k++) {
        const j = (k + 1) % 3;
        if (count.get(key(m.tris[t * 3 + k], m.tris[t * 3 + j])) === 1) idx.push(top[k], top[j], bot[j], top[k], bot[j], bot[k]);
      }
    }
    if (!idx.length) return null;
    const nv = node.length;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
    const disp = new THREE.BufferAttribute(new Float32Array(nv * 3), 3);
    const val = new THREE.BufferAttribute(new Float32Array(nv).fill(NONE), 1);
    disp.setUsage(THREE.DynamicDrawUsage);
    val.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("disp", disp);
    g.setAttribute("val", val);
    g.setAttribute("pid", new THREE.BufferAttribute(new Float32Array(ids), 1));
    g.setIndex(nv > 65535 ? new THREE.BufferAttribute(new Uint32Array(idx), 1) : new THREE.BufferAttribute(new Uint16Array(idx), 1));
    const mesh = new THREE.Mesh(g, this.solidMat);
    mesh.name = "profiles";
    mesh.frustumCulled = false;
    const pick = new THREE.Mesh(g, this.pickMat);
    pick.frustumCulled = false;
    return { mesh, pick, node: new Uint32Array(node) };
  }

  /** Copies nodal displacements and values to the profile vertices. */
  private syncSolid() {
    const sd = this.solid;
    if (!sd || !this.disp || !this.val) return;
    const g = sd.mesh.geometry;
    const da = g.getAttribute("disp") as THREE.BufferAttribute;
    const va = g.getAttribute("val") as THREE.BufferAttribute;
    const d = this.disp.array as Float32Array;
    const v = this.val.array as Float32Array;
    const od = da.array as Float32Array;
    const ov = va.array as Float32Array;
    const nd = sd.node;
    for (let i = 0; i < nd.length; i++) {
      const n = nd[i];
      od[i * 3] = d[n * 3];
      od[i * 3 + 1] = d[n * 3 + 1];
      od[i * 3 + 2] = d[n * 3 + 2];
      ov[i] = v[n];
    }
    da.needsUpdate = va.needsUpdate = true;
  }

  private buildPick() {
    const m = this.mesh!;
    if (m.tris.length) {
      // non-indexed copy with element id per vertex
      const nt = m.triElem.length;
      const g = new THREE.BufferGeometry();
      const pos = new Float32Array(nt * 9);
      const ids = new Float32Array(nt * 3);
      for (let t = 0; t < nt; t++) {
        for (let k = 0; k < 3; k++) {
          const n = m.tris[t * 3 + k];
          pos.set(m.positions.subarray(n * 3, n * 3 + 3), (t * 3 + k) * 3);
          ids[t * 3 + k] = m.triElem[t] + 1;
        }
      }
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("disp", new THREE.BufferAttribute(new Float32Array(nt * 9), 3));
      g.setAttribute("pid", new THREE.BufferAttribute(ids, 1));
      const pm = new THREE.Mesh(g, this.pickMat);
      pm.frustumCulled = false;
      pm.name = "pick-tris";
      this.pickScene.add(pm);
    }
    if (this.fat) {
      const pf = new THREE.Mesh(this.fat.geometry, this.fatPickMat);
      pf.frustumCulled = false;
      this.pickScene.add(pf);
    }
    if (this.solid) this.pickScene.add(this.solid.pick);
  }

  private syncPickDisp() {
    const m = this.mesh;
    const pm = this.pickScene.getObjectByName("pick-tris") as THREE.Mesh | undefined;
    if (!m || !pm || !this.disp) return;
    const a = pm.geometry.getAttribute("disp") as THREE.BufferAttribute;
    const d = this.disp.array as Float32Array;
    const out = a.array as Float32Array;
    for (let i = 0; i < m.tris.length; i++) {
      const n = m.tris[i];
      out[i * 3] = d[n * 3];
      out[i * 3 + 1] = d[n * 3 + 1];
      out[i * 3 + 2] = d[n * 3 + 2];
    }
    a.needsUpdate = true;
  }

  /** Displacements per node (null = undeformed). */
  setDisplacement(d: Float32Array | null) {
    if (!this.disp || !this.mesh) return;
    const arr = this.disp.array as Float32Array;
    if (d && d.length === arr.length) arr.set(d);
    else arr.fill(0);
    this.disp.needsUpdate = true;
    if (this.fat) {
      const g = this.fat.geometry;
      const da = g.getAttribute("da") as THREE.BufferAttribute;
      const db = g.getAttribute("db") as THREE.BufferAttribute;
      const L = this.mesh.lines;
      for (let s = 0; s < L.length / 2; s++) {
        for (let k = 0; k < 4; k++) {
          for (let c = 0; c < 3; c++) {
            (da.array as Float32Array)[(s * 4 + k) * 3 + c] = arr[L[2 * s] * 3 + c];
            (db.array as Float32Array)[(s * 4 + k) * 3 + c] = arr[L[2 * s + 1] * 3 + c];
          }
        }
      }
      da.needsUpdate = db.needsUpdate = true;
    }
    this.syncPickDisp();
    this.syncSolid();
    this.requestRender();
  }

  /** Nodal values for contours (null = no contour). */
  setField(values: Float32Array | null, min: number, max: number) {
    if (!this.val || !this.mesh) return;
    const arr = this.val.array as Float32Array;
    const on = !!values && values.length === arr.length;
    if (on) {
      for (let i = 0; i < arr.length; i++) {
        const v = values![i];
        arr[i] = Number.isFinite(v) ? v : NONE;
      }
    } else arr.fill(NONE);
    this.val.needsUpdate = true;
    for (const mat of [this.surfMat, this.fatMat]) {
      mat.uniforms.useField.value = on ? 1 : 0;
      mat.uniforms.vmin.value = min;
      mat.uniforms.vmax.value = max;
    }
    if (this.fat) {
      const g = this.fat.geometry;
      const va = g.getAttribute("va") as THREE.BufferAttribute;
      const vb = g.getAttribute("vb") as THREE.BufferAttribute;
      const L = this.mesh.lines;
      for (let s = 0; s < L.length / 2; s++) {
        for (let k = 0; k < 4; k++) {
          (va.array as Float32Array)[s * 4 + k] = arr[L[2 * s]];
          (vb.array as Float32Array)[s * 4 + k] = arr[L[2 * s + 1]];
        }
      }
      va.needsUpdate = vb.needsUpdate = true;
    }
    this.syncSolid();
    this.requestRender();
  }

  /** Magnification of the displacements. */
  setScale(s: number) {
    this.scale = s;
    this.surfMat.uniforms.scale.value = s;
    this.updateVisibility();
    this.requestRender();
  }

  /** Animate a mode (the scale oscillates) or stop. */
  animateMode(on: boolean) {
    if (on) {
      this.animating = { kind: "mode", base: this.scale, t0: performance.now() };
      this.requestRender();
    } else if (this.animating) {
      const base = this.animating.base;
      this.animating = null;
      this.setScale(base);
    }
  }

  /** Supports and loads. `fmt` writes load values (magnitude, moment?) as labels; null = no labels. */
  setSymbols(sym: Symbols | null, fmt: ((v: number, moment: boolean) => string) | null = null) {
    for (const c of [...this.symbolGroup.children]) {
      this.symbolGroup.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
      if (c instanceof THREE.Sprite) {
        c.material.map?.dispose();
        c.material.dispose();
      }
    }
    if (!sym || !this.mesh) {
      this.requestRender();
      return;
    }
    const P = this.mesh.positions;
    const size = this.radius * 0.035;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    const p = new THREE.Vector3();
    if (sym.bcNodes.length) this.addSupports(sym, size);
    if (sym.loadNodes.length) {
      // Arrows with the tip at the point of application and the shaft upstream, in the load's direction.
      // The geometry lies along +y with the tip at the origin; the length (the scale in y) follows
      // the magnitude, relative to the largest load of the same kind (at least 45 %).
      const len = size * 3.2;
      const headLen = size * 0.9;
      const head = new THREE.ConeGeometry(size * 0.32, headLen, 14);
      head.rotateZ(Math.PI);
      head.translate(0, headLen / 2, 0);
      const shaft = new THREE.CylinderGeometry(size * 0.085, size * 0.085, 1, 8);
      shaft.translate(0, 0.5, 0);
      // moment: double arrow (the second tip behind the first)
      const head2 = head.clone().translate(0, headLen * 0.75, 0);
      const forceMat = new THREE.MeshBasicMaterial({ color: 0xe0559a, depthTest: false, transparent: true });
      const momentMat = new THREE.MeshBasicMaterial({ color: 0xa371f7, depthTest: false, transparent: true });
      const items: { n: number; d: THREE.Vector3; mag: number; moment: boolean }[] = [];
      for (let i = 0; i < sym.loadNodes.length; i++) {
        const f = new THREE.Vector3(sym.loads[i * 6], sym.loads[i * 6 + 1], sym.loads[i * 6 + 2]);
        const mo = new THREE.Vector3(sym.loads[i * 6 + 3], sym.loads[i * 6 + 4], sym.loads[i * 6 + 5]);
        if (f.lengthSq() > 0) items.push({ n: sym.loadNodes[i], mag: f.length(), d: f.normalize(), moment: false });
        if (mo.lengthSq() > 0) items.push({ n: sym.loadNodes[i], mag: mo.length(), d: mo.normalize(), moment: true });
      }
      for (const moment of [false, true]) {
        const sel = items.filter((x) => x.moment === moment);
        if (!sel.length) continue;
        const max = Math.max(...sel.map((x) => x.mag));
        const parts: [THREE.BufferGeometry, (l: number) => number, number][] = [
          [head, () => 1, 0],
          [shaft, (l) => l - headLen, headLen],
          ...(moment ? [[head2, () => 1, 0] as [THREE.BufferGeometry, (l: number) => number, number]] : []),
        ];
        for (const [geo, stretch, offset] of parts) {
          const inst = new THREE.InstancedMesh(geo, moment ? momentMat : forceMat, sel.length);
          sel.forEach((it, i) => {
            const l = len * (0.45 + 0.55 * (it.mag / max));
            // +y in the geometry points upstream, i.e. towards −d
            q.setFromUnitVectors(up, it.d.clone().negate());
            p.fromArray(P, it.n * 3).addScaledVector(it.d, -offset);
            m.compose(p, q, new THREE.Vector3(1, Math.max(stretch(l), 0), 1));
            inst.setMatrixAt(i, m);
          });
          inst.frustumCulled = false;
          inst.renderOrder = 4;
          this.symbolGroup.add(inst);
        }
        if (fmt) {
          for (const g of loadGroups(sel, P, this.radius)) {
            const at = g.center.clone().addScaledVector(g.d, -(len + size * 0.9));
            this.symbolGroup.add(this.label(`${g.count > 1 ? "Σ " : ""}${fmt(g.total, moment)}`, at, size * 0.95, moment ? "#a371f7" : "#e0559a"));
          }
        }
      }
    }
    this.updateVisibility();
    this.requestRender();
  }

  /**
   * Support symbols: fixed (triangle on a slab), pinned (triangle on a line) and
   * roller (triangle on wheels, one per locked translation, oriented along the locked
   * direction). The mask has U1–U3 in bits 0–2 and UR1–UR3 in bits 3–5.
   */
  private addSupports(sym: Symbols, s: number) {
    const P = this.mesh!.positions;
    const is2d = this.is2d;
    const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
    const up = axes[1];
    const nt = is2d ? 2 : 3;
    // nodes in beams/trusses can have rotations; continuum nodes only have translations
    const lineNode = new Set<number>(this.mesh!.lines);
    const fixT: THREE.Matrix4[] = [], pinT: THREE.Matrix4[] = [], rollT: THREE.Matrix4[] = [], rotT: THREE.Matrix4[] = [];
    const q = new THREE.Quaternion();
    const p = new THREE.Vector3();
    const one = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < sym.bcNodes.length; i++) {
      const n = sym.bcNodes[i];
      const mask = sym.bcMasks[i];
      p.fromArray(P, n * 3);
      const locked = [0, 1, 2].slice(0, nt).filter((a) => mask & (1 << a));
      const rotAll = is2d ? (mask & 32) !== 0 : (mask & 56) === 56;
      if (locked.length === nt) {
        q.setFromUnitVectors(up, up);
        (rotAll || !lineNode.has(n) ? fixT : pinT).push(new THREE.Matrix4().compose(p, q, one));
      } else if (locked.length) {
        for (const a of locked) {
          q.setFromUnitVectors(up, axes[a]);
          rollT.push(new THREE.Matrix4().compose(p, q, one));
        }
      } else if (mask & 56) {
        rotT.push(new THREE.Matrix4().compose(p, q.identity(), one));
      }
    }
    // the geometry in a local frame: the tip at the node, the symbol downward along −y
    const depth = is2d ? s * 0.05 : s * 1.3;
    const tri = new THREE.ConeGeometry(s * 0.55, s, is2d ? 3 : 4);
    if (!is2d) tri.rotateY(Math.PI / 4);
    tri.translate(0, -s / 2, 0);
    const slab = new THREE.BoxGeometry(s * 1.4, s * 0.24, is2d ? s * 0.05 : s * 1.4).translate(0, -s - s * 0.12, 0);
    const line = new THREE.BoxGeometry(s * 1.4, s * 0.07, depth).translate(0, -s - s * 0.035, 0);
    const r = s * 0.16;
    const wheelPos = is2d ? [[-0.3, 0], [0.3, 0]] : [[-0.3, -0.3], [0.3, -0.3], [-0.3, 0.3], [0.3, 0.3]];
    const wheels = wheelPos.map(([x, z]) => new THREE.SphereGeometry(r, 14, 10).translate(x * s, -s - r, z * s));
    const rollLine = new THREE.BoxGeometry(s * 1.4, s * 0.07, depth).translate(0, -s - 2 * r - s * 0.035, 0);
    const mat = (c: number) => new THREE.MeshBasicMaterial({ color: c });
    const fixMat = mat(0xf08c2e), pinMat = mat(0xf2b84b), rollMat = mat(0x4fc3d9), rotMat = mat(0xa371f7);
    const add = (geos: THREE.BufferGeometry[], material: THREE.Material, ts: THREE.Matrix4[]) => {
      if (!ts.length) return;
      for (const g of geos) {
        const inst = new THREE.InstancedMesh(g, material, ts.length);
        ts.forEach((t, i) => inst.setMatrixAt(i, t));
        inst.frustumCulled = false;
        this.symbolGroup.add(inst);
      }
    };
    add([tri, slab], fixMat, fixT);
    add([tri, line], pinMat, pinT);
    add([tri, ...wheels, rollLine], rollMat, rollT);
    // only rotations locked: a small cube at the node
    add([new THREE.BoxGeometry(s * 0.45, s * 0.45, s * 0.45)], rotMat, rotT);
  }

  /** Text label that is always on top (height `h` in the model's length unit). */
  private label(text: string, at: THREE.Vector3, h: number, color: string): THREE.Sprite {
    const px = 56;
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d")!;
    const font = `600 ${px}px system-ui, "Segoe UI", sans-serif`;
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + px * 0.7;
    c.width = w;
    c.height = Math.ceil(px * 1.45);
    ctx.font = font;
    ctx.fillStyle = this.opts.dark ? "rgba(17,19,25,0.82)" : "rgba(255,255,255,0.88)";
    ctx.beginPath();
    ctx.roundRect(0, 0, c.width, c.height, px * 0.3);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, px * 0.35, c.height / 2 + 2);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set((h * c.width) / c.height, h, 1);
    sp.position.copy(at);
    sp.renderOrder = 8;
    return sp;
  }

  /** Highlights an element (null = no highlight). */
  setHighlight(h: Hit | null) {
    for (const c of [...this.highlight.children]) {
      this.highlight.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
    const m = this.mesh;
    if (h && m && this.pos && this.disp) {
      if (h.kind === "element") {
        const tri: number[] = [];
        for (let t = 0; t < m.triElem.length; t++) if (m.triElem[t] === h.index) tri.push(m.tris[t * 3], m.tris[t * 3 + 1], m.tris[t * 3 + 2]);
        const seg: number[] = [];
        for (let s = 0; s < m.lineElem.length; s++) if (m.lineElem[s] === h.index) seg.push(m.lines[2 * s], m.lines[2 * s + 1]);
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", this.pos);
        g.setAttribute("disp", this.disp);
        if (tri.length) {
          g.setIndex(tri);
          const o = new THREE.Mesh(g, this.hiMat);
          o.frustumCulled = false;
          o.renderOrder = 5;
          this.highlight.add(o);
        }
        if (seg.length) {
          const gl = new THREE.BufferGeometry();
          gl.setAttribute("position", this.pos);
          gl.setAttribute("disp", this.disp);
          gl.setIndex(seg);
          const o = new THREE.LineSegments(gl, this.hiMat);
          o.frustumCulled = false;
          this.highlight.add(o);
        }
      } else {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", this.pos);
        g.setAttribute("disp", this.disp);
        g.setIndex([h.index]);
        const mat = this.pointMat.clone();
        mat.vertexShader = lineVert.replace("gl_Position =", "gl_PointSize = 14.0;\n  gl_Position =");
        mat.uniforms = { scale: this.surfMat.uniforms.scale, color: { value: new THREE.Color(0xffd84d) }, opacity: { value: 1 } };
        mat.depthTest = false;
        const o = new THREE.Points(g, mat);
        o.frustumCulled = false;
        o.renderOrder = 6;
        this.highlight.add(o);
      }
    }
    this.requestRender();
  }

  /** Marks nodes (preview of a selection; null or empty = none). */
  setMarked(nodes: number[] | null) {
    for (const c of [...this.marks.children]) {
      this.marks.remove(c);
      (c as THREE.Points).geometry?.dispose();
    }
    if (nodes?.length && this.mesh && this.pos && this.disp) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", this.pos);
      g.setAttribute("disp", this.disp);
      g.setIndex(nodes);
      const mat = this.pointMat.clone();
      mat.vertexShader = lineVert.replace("gl_Position =", "gl_PointSize = 9.0;\n  gl_Position =");
      mat.uniforms = { scale: this.surfMat.uniforms.scale, color: { value: new THREE.Color(0xff8a3d) }, opacity: { value: 1 } };
      mat.depthTest = false;
      const o = new THREE.Points(g, mat);
      o.frustumCulled = false;
      o.renderOrder = 6;
      this.marks.add(o);
    }
    this.requestRender();
  }

  set2D(on: boolean) {
    this.is2d = on;
    this.controls.enableRotate = !on;
    this.controls.mouseButtons = {
      LEFT: on ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.touches = { ONE: on ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    if (on) this.view("front");
    this.requestRender();
  }

  get mode2d() {
    return this.is2d;
  }

  view(dir: ViewDir) {
    const [d, up] = DIRS[dir];
    const dist = this.radius * 4;
    this.camera.up.copy(up);
    this.camera.position.copy(this.center).addScaledVector(d.clone().normalize(), dist);
    this.camera.near = -dist * 10;
    this.camera.far = dist * 10;
    this.controls.target.copy(this.center);
    this.fit();
  }

  fit() {
    this.autoFit = true;
    this.camera.zoom = 1;
    this.resize();
    // keep the viewing direction: an earlier fit may have moved the camera and target sideways
    this.camera.position.sub(this.controls.target).add(this.center);
    this.controls.target.copy(this.center);
    this.camera.lookAt(this.center);
    this.camera.updateMatrixWorld();
    // zoom so that what is drawn (seen from the camera) fills the window: the nodes and the
    // profile/thickness solids, in the deformed shape and, where shown, the undeformed one
    const inv = this.camera.matrixWorldInverse;
    const c = new THREE.Vector3().copy(this.center).applyMatrix4(inv);
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    const e = inv.elements;
    const scales = this.animating ? [this.animating.base, -this.animating.base] : [this.scale];
    if (!scales.includes(0) && this.opts.undeformed) scales.push(0);
    const span = (P: ArrayLike<number>, D: ArrayLike<number> | null, n: number, used: Uint8Array | null) => {
      for (const s of D ? scales : [0]) {
        for (let i = 0; i < n; i++) {
          if (used && !used[i]) continue;
          const x = P[i * 3] + (D ? s * D[i * 3] : 0);
          const y = P[i * 3 + 1] + (D ? s * D[i * 3 + 1] : 0);
          const z = P[i * 3 + 2] + (D ? s * D[i * 3 + 2] : 0);
          const px = e[0] * x + e[4] * y + e[8] * z + e[12];
          const py = e[1] * x + e[5] * y + e[9] * z + e[13];
          if (px < x0) x0 = px;
          if (px > x1) x1 = px;
          if (py < y0) y0 = py;
          if (py > y1) y1 = py;
        }
      }
    };
    if (this.mesh) span(this.mesh.positions, this.disp ? (this.disp.array as Float32Array) : null, this.used.length, this.used);
    if (this.solid && this.opts.profiles) {
      const g = this.solid.mesh.geometry;
      span(g.getAttribute("position").array, g.getAttribute("disp").array, this.solid.node.length, null);
    }
    if (!(x1 >= x0)) {
      const b = this.box;
      span([b.min.x, b.min.y, b.min.z, b.max.x, b.min.y, b.min.z, b.min.x, b.max.y, b.min.z, b.max.x, b.max.y, b.min.z, b.min.x, b.min.y, b.max.z, b.max.x, b.min.y, b.max.z, b.min.x, b.max.y, b.max.z, b.max.x, b.max.y, b.max.z], null, 8, null);
    }
    const ex = (x1 - x0) / 2;
    const ey = (y1 - y0) / 2;
    // centre the view on that rectangle
    const mw = this.camera.matrixWorld;
    const off = new THREE.Vector3()
      .setFromMatrixColumn(mw, 0)
      .multiplyScalar((x0 + x1) / 2 - c.x)
      .addScaledVector(new THREE.Vector3().setFromMatrixColumn(mw, 1), (y0 + y1) / 2 - c.y);
    this.camera.position.add(off);
    this.controls.target.add(off);
    // the view tools float over the right edge: fit the model into the area left of them
    const w = Math.max(this.canvas.clientWidth, 1);
    const tools = this.canvas.parentElement?.querySelector<HTMLElement>(".vp-tools");
    const inset = Math.min(tools ? tools.offsetWidth + 20 : 0, w * 0.4);
    if (ex > 0 || ey > 0) {
      const z = Math.min((this.camera.right * (w - inset)) / w / Math.max(ex, 1e-12), this.camera.top / Math.max(ey, 1e-12)) * 0.82;
      this.camera.zoom = Math.min(Math.max(z, 0.05), 50);
      this.camera.updateProjectionMatrix();
    }
    if (inset > 0) {
      // move the camera sideways so that the model is centred in that area
      const perPx = (this.camera.right - this.camera.left) / this.camera.zoom / w;
      const shift = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0).multiplyScalar((inset / 2) * perPx);
      this.camera.position.add(shift);
      this.controls.target.add(shift);
    }
    this.controls.update();
    this.requestRender();
  }

  private resize() {
    const el = this.canvas.parentElement ?? this.canvas;
    const w = Math.max(el.clientWidth, 1);
    const h = Math.max(el.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    const aspect = w / h;
    const r = this.radius * 1.15;
    const hh = aspect >= 1 ? r : r / aspect;
    this.camera.left = -hh * aspect;
    this.camera.right = hh * aspect;
    this.camera.top = hh;
    this.camera.bottom = -hh;
    this.camera.updateProjectionMatrix();
    const pr = this.renderer.getPixelRatio();
    (this.fatMat.uniforms.resolution.value as THREE.Vector2).set(w * pr, h * pr);
    this.requestRender();
  }

  requestRender() {
    if (!this.frameReq) this.frameReq = requestAnimationFrame(this.render);
  }

  private render = () => {
    this.frameReq = 0;
    if (this.animating) {
      const t = (performance.now() - this.animating.t0) / 1000;
      this.surfMat.uniforms.scale.value = this.animating.base * Math.sin(t * Math.PI * 1.4);
      this.frameReq = requestAnimationFrame(this.render);
    }
    const r = this.renderer;
    // setViewport/setScissor take CSS pixels (the renderer multiplies by pixelRatio)
    const w = r.domElement.clientWidth;
    const h = r.domElement.clientHeight;
    r.setViewport(0, 0, w, h);
    r.setScissorTest(false);
    r.clear();
    r.render(this.scene, this.camera);
    // axis triad in the lower left corner
    const s = 84;
    r.clearDepth();
    r.setScissorTest(true);
    r.setScissor(8, 8, s, s);
    r.setViewport(8, 8, s, s);
    this.triad.camera.position.copy(this.camera.position).sub(this.controls.target).normalize().multiplyScalar(5);
    this.triad.camera.up.copy(this.camera.up);
    this.triad.camera.lookAt(0, 0, 0);
    r.render(this.triad.scene, this.triad.camera);
    r.setScissorTest(false);
  };

  private onDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY, t: performance.now(), touch: e.pointerType !== "mouse" };
  };

  private onUp = (e: PointerEvent) => {
    const d = this.down;
    this.down = null;
    if (!d || e.button > 0) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > (d.touch ? 10 : 4) || performance.now() - d.t > 600) return;
    this.onPick(this.pick(e.clientX, e.clientY, d.touch));
  };

  /** GPU picking: element id as color in a small hidden image around the cursor. */
  pick(clientX: number, clientY: number, touch: boolean): Hit | null {
    if (!this.mesh) return null;
    const rect = this.canvas.getBoundingClientRect();
    const pr = this.renderer.getPixelRatio();
    const W = Math.round(rect.width * pr);
    const H = Math.round(rect.height * pr);
    const half = Math.round((touch ? 22 : 7) * pr);
    const size = half * 2 + 1;
    const px = Math.round((clientX - rect.left) * pr);
    const py = Math.round((clientY - rect.top) * pr);
    this.pickTarget.setSize(size, size);
    this.camera.setViewOffset(W, H, px - half, py - half, size, size);
    const r = this.renderer;
    const old = r.getClearColor(new THREE.Color());
    const oldA = r.getClearAlpha();
    r.setRenderTarget(this.pickTarget);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.render(this.pickScene, this.camera);
    const buf = new Uint8Array(size * size * 4);
    r.readRenderTargetPixels(this.pickTarget, 0, 0, size, size, buf);
    r.setRenderTarget(null);
    r.setClearColor(old, oldA);
    this.camera.clearViewOffset();
    let best = -1;
    let bestD = Infinity;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const id = buf[i] + buf[i + 1] * 256 + buf[i + 2] * 65536;
        if (id === 0) continue;
        const dd = (x - half) ** 2 + (y - half) ** 2;
        if (dd < bestD) {
          bestD = dd;
          best = id - 1;
        }
      }
    }
    this.requestRender();
    if (best < 0) return null;
    if (this.opts.pickMode === "element") return { kind: "element", index: best };
    return this.nearestNode(best, clientX - rect.left, clientY - rect.top, rect.width, rect.height);
  }

  private nearestNode(elem: number, x: number, y: number, w: number, h: number): Hit | null {
    const m = this.mesh!;
    const cand = new Set<number>();
    for (let t = 0; t < m.triElem.length; t++) if (m.triElem[t] === elem) for (let k = 0; k < 3; k++) cand.add(m.tris[t * 3 + k]);
    for (let s = 0; s < m.lineElem.length; s++) if (m.lineElem[s] === elem) cand.add(m.lines[2 * s]).add(m.lines[2 * s + 1]);
    const v = new THREE.Vector3();
    const d = this.disp!.array as Float32Array;
    let best = -1;
    let bd = Infinity;
    for (const n of cand) {
      v.set(m.positions[n * 3] + this.scale * d[n * 3], m.positions[n * 3 + 1] + this.scale * d[n * 3 + 1], m.positions[n * 3 + 2] + this.scale * d[n * 3 + 2]).project(this.camera);
      const sx = ((v.x + 1) / 2) * w;
      const sy = ((1 - v.y) / 2) * h;
      const dd = (sx - x) ** 2 + (sy - y) ** 2;
      if (dd < bd) {
        bd = dd;
        best = n;
      }
    }
    return best >= 0 ? { kind: "node", index: best } : null;
  }

  /** Measures the frame time: `frames` frames while the camera rotates one turn (ms per frame). */
  benchmark(frames: number): number {
    const gl = this.renderer.getContext();
    const target = this.controls.target.clone();
    const start = this.camera.position.clone().sub(target);
    const px = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) {
      const a = (i / frames) * Math.PI * 2;
      this.camera.position.copy(start).applyAxisAngle(this.camera.up, a).add(target);
      this.camera.lookAt(target);
      this.render();
    }
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const ms = (performance.now() - t0) / frames;
    this.camera.position.copy(start).add(target);
    this.camera.lookAt(target);
    this.controls.update();
    this.requestRender();
    return ms;
  }

  /** Number of triangles and line segments drawn. */
  get stats() {
    return { triangles: this.mesh?.triElem.length ?? 0, segments: (this.mesh?.edges.length ?? 0) / 2 + (this.mesh?.lineElem.length ?? 0) };
  }

  screenshot(): string {
    this.render();
    return this.canvas.toDataURL("image/png");
  }
}
