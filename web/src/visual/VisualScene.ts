// The Visualization tab: the model drawn as a structure (beams with their real
// profiles, shells with their thickness, solids), with deformation, a rendering
// mode (sky, ground, shadows), section cuts with section force diagrams and
// support reactions. A general version of the course's bridge viewer: everything
// is taken from the model (`app/session/src/visual.rs`).

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Engine, MeshData, StepResultInfo, Vec3, VisCut, VisDiagram, VisFrame, VisMember, VisModel, VisPart, VisProfile } from "../engine/Engine";
import { convertValue, type ConvQ } from "../ui/units";

export interface VisualUnits {
  /** The model's unit per quantity (null = no unit system: values are shown as they are). */
  model: (q: ConvQ) => string | null;
  /** The chosen display unit per quantity. */
  show: (q: ConvQ) => string | null;
  system: string | undefined;
}

export interface VisualInput {
  engine: Engine;
  id: number;
  name: string;
  mesh: MeshData;
  vm: VisModel;
  results: StepResultInfo[];
  units: VisualUnits;
  dark: boolean;
  /** Writes a PNG (data URL); returns false if the user cancelled. */
  savePng: (dataUrl: string) => Promise<void>;
}

const V3 = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const V2 = (x: number, y: number) => new THREE.Vector2(x, y);
const vec = (a: ArrayLike<number>, i = 0) => V3(a[i], a[i + 1], a[i + 2]);

const PALETTE = [0x3d6fb4, 0xc8473f, 0xe0a52e, 0x3f9a6b, 0x8a5fc2, 0xd9772b, 0x9aa7b5, 0x2aa7b8, 0x6f7f90, 0xb5557a, 0x7d9a3a, 0x4f5fa8];
const SUPPORT = 0x2aa7b8;
// colours of the course figures: forces and moments blue, axes dark red, the cut face light grey
const FORCE = 0x2f5597, AXIS = 0xa3283f, FACE = 0xe7e6e6, FACE_EDGE = 0x595959;
const SCHEMES: Record<string, number> = { grey: 0x7c868f, green: 0x3f5f52, white: 0xe9ebec, red: 0x9e3a2f, corten: 0x7a4a33 };
const sym = (s: string, i?: string) => `<i>${s}</i>${i ? `<sub>${i}</sub>` : ""}`;

/** A drawn object whose vertices follow the displacements. */
interface Rec {
  kind: "member" | "body" | "shell" | "part" | "support";
  group: Group;
  mesh: THREE.Mesh;
  edges: THREE.LineSegments | null;
  member?: number;
  body?: number;
  support?: number;
  /** Per vertex of the mesh and of the edges: how its displacement is found. */
  map: VertMap[];
  base: Float32Array[];
  disp: (Float32Array | null)[];
  attrs: THREE.BufferAttribute[];
}

/** Member vertices: path segment and position along it; node vertices: node index. */
type VertMap = { seg: Int32Array; xi: Float32Array } | { node: Int32Array };

interface Group {
  part: number;
  name: string;
  desc: string;
  color: number;
  g: THREE.Group;
  mat: THREE.MeshStandardMaterial;
  lineMat: THREE.LineBasicMaterial;
  recs: Rec[];
  visible: boolean;
  isSupport?: boolean;
}

interface MemberPath {
  m: VisMember;
  pts: THREE.Vector3[];
  t: THREE.Vector3[];
  n1: THREE.Vector3[];
  n2: THREE.Vector3[];
  beam: boolean;
}

interface Sel {
  kind: "member" | "body";
  index: number;
  rec: Rec;
  recs: Rec[];
  v: number;
  vmin: number;
  vmax: number;
  plane: THREE.Plane;
  old: [THREE.Material, THREE.Material | null][];
  cap: THREE.Object3D;
  disc: THREE.Mesh;
  arrows: THREE.Group;
  tips: { pos: THREE.Vector3; html: string; cls: string }[];
  dirty: boolean;
  /** Bodies: the diagram along the axis and the latest cut. */
  diag?: VisDiagram | null;
  cut?: VisCut | null;
  axis?: THREE.Vector3;
  axisMode?: "auto" | "x" | "y" | "z";
  pending?: boolean;
  want?: number | null;
}

interface SupportObj {
  index: number;
  open: boolean;
  recs: Rec[];
  g: THREE.Group | null;
  tips: [THREE.Vector3, HTMLDivElement][];
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, html?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function niceTicks(a: number, b: number, n: number) {
  const raw = (b - a) / n;
  if (!(raw > 0)) return [a];
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const st = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(a / st) * st; v <= b + 1e-9 * Math.abs(b); v += st) out.push(Math.round(v / st) * st);
  return out;
}

/** Member axes as in the course: x along the member, z downwards, y to the left of the positive section. */
function localAxes(t: THREE.Vector3) {
  let z = V3(0, -1, 0).addScaledVector(t, t.y);
  if (z.length() < 0.2) z = V3(1, 0, 0).addScaledVector(t, -t.x);
  z.normalize();
  return { x: t.clone(), y: z.clone().cross(t), z };
}

function arrow(from: THREE.Vector3, to: THREE.Vector3, color: number, r: number, heads = 1) {
  const g = new THREE.Group(), m = new THREE.MeshBasicMaterial({ color }), d = to.clone().sub(from), len = d.length();
  if (len < 1e-12) return g;
  const h = Math.min(7 * r, (0.42 * len) / heads), shaftLen = Math.max(1e-6, len - h * (heads === 2 ? 1.75 : 1));
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(r, r, shaftLen, 14), m);
  shaft.position.y = shaftLen / 2;
  g.add(shaft);
  for (let k = 0; k < heads; k++) {
    const head = new THREE.Mesh(new THREE.ConeGeometry(r * 3, h, 18), m);
    head.position.y = len - h / 2 - k * 0.75 * h;
    g.add(head);
  }
  g.position.copy(from);
  g.quaternion.setFromUnitVectors(V3(0, 1, 0), d.normalize());
  return g;
}

function hermite(xi: number) {
  const x2 = xi * xi, x3 = x2 * xi;
  return [1 - 3 * x2 + 2 * x3, xi - 2 * x2 + x3, 3 * x2 - 2 * x3, -x2 + x3];
}

const CSS_ID = "vis-style";

export class VisualScene {
  private root: HTMLDivElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private hemi: THREE.HemisphereLight;
  private sun: THREE.DirectionalLight;
  private fill: THREE.DirectionalLight;
  private groups: Group[] = [];
  private paths: MemberPath[] = [];
  private supports: SupportObj[] = [];
  private supGroup: Group | null = null;
  private center = new THREE.Vector3();
  private box = new THREE.Box3();
  private R = 1;
  private yGround = 0;
  private sky: THREE.Mesh;
  private ground: THREE.Mesh;
  private envTex: THREE.Texture;
  private raf = 0;
  private ro: ResizeObserver;
  private lastK: number | null = null;
  private step = 0;
  private frame = 0;
  private U: Float32Array | null = null;
  private F: VisFrame | null = null;
  private maxU = { u: 0, node: -1 };
  private autoScale = 1;
  private sel: Sel | null = null;
  private pick: { kind: "support"; index: number } | { kind: "info"; html: string } | null = null;
  private fly: { t0: number; p0: THREE.Vector3; q0: THREE.Vector3; p1: THREE.Vector3; q1: THREE.Vector3 } | null = null;
  private ray = new THREE.Raycaster();
  private down: [number, number] | null = null;
  private dragging = false;
  private plotW = 300;
  private plotH = 150;
  private reactGroup = new THREE.Group();
  private cutGroup = new THREE.Group();
  private ui: Record<string, HTMLElement> = {};
  private opt = { def: true, anim: true, scale: 25, transp: false, edges: true, render: false, scheme: "parts", sun: 55 };
  private disposed = false;
  private hasResults: boolean;
  private is2d: boolean;

  constructor(host: HTMLElement, private inp: VisualInput) {
    this.hasResults = inp.results.length > 0;
    VisualScene.injectCss();
    this.root = el("div", `vis${inp.dark ? " dark" : ""}`);
    host.appendChild(this.root);
    const dark = inp.dark;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.root.appendChild(this.renderer.domElement);
    this.scene.background = new THREE.Color(dark ? 0x12161b : 0xeef1f4);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.hemi = new THREE.HemisphereLight(0xffffff, dark ? 0x30363f : 0x8a94a0, 2.9);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.3);
    this.fill = new THREE.DirectionalLight(0xffffff, 0.9);
    this.scene.add(this.hemi, this.sun, this.fill, this.sun.target, this.reactGroup, this.cutGroup);

    const pos = inp.mesh.positions;
    const box = new THREE.Box3();
    for (let i = 0; i < pos.length; i += 3) box.expandByPoint(vec(pos, i));
    box.getCenter(this.center);
    this.box = box;
    this.R = Math.max(1e-9, box.getSize(new THREE.Vector3()).length() / 2);
    this.is2d = box.max.z - box.min.z < 1e-9 * this.R && !inp.mesh.profiles?.threeD;

    this.buildUi();
    this.buildGeometry();
    this.buildSupportsFromSymbols();
    // the environment for the rendering mode, scaled to the model
    const R = this.R;
    this.yGround = box.min.y - 0.06 * R;
    this.sky = this.skyDome(16 * R);
    this.sky.position.copy(this.center);
    this.sky.visible = false;
    this.scene.add(this.sky);
    this.ground = new THREE.Mesh(new THREE.CircleGeometry(16 * R, 64), new THREE.MeshStandardMaterial({ color: 0x7f8a6e, roughness: 1, metalness: 0 }));
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.set(this.center.x, this.yGround, this.center.z);
    this.ground.receiveShadow = true;
    this.ground.visible = false;
    this.scene.add(this.ground);
    const envScene = new THREE.Scene();
    envScene.add(this.skyDome(50));
    const s = new THREE.Mesh(new THREE.SphereGeometry(6, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    s.position.set(20, 30, 20);
    envScene.add(s);
    this.envTex = new THREE.PMREMGenerator(this.renderer).fromScene(envScene, 0.03).texture;
    this.sun.shadow.mapSize.set(4096, 4096);
    Object.assign(this.sun.shadow.camera, { left: -1.4 * R, right: 1.4 * R, top: 1.4 * R, bottom: -1.4 * R, near: 0.05 * R, far: 8 * R });
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.002 * R;
    this.sun.target.position.copy(this.center);

    this.camera.near = this.R / 500;
    this.camera.far = this.R * 60;
    this.camera.updateProjectionMatrix();
    this.applyOpts();
    this.applyRender();
    this.setView(this.is2d ? "side" : "iso");

    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", this.onDownCapture, true);
    dom.addEventListener("pointermove", this.onMove);
    dom.addEventListener("pointerup", this.onUpCapture, true);
    dom.addEventListener("pointerdown", this.onDown);
    dom.addEventListener("pointerup", this.onUp);
    window.addEventListener("keydown", this.onKey);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.root);
    this.resize();
    if (this.hasResults) {
      const last = inp.results.length - 1;
      this.setStep(last, inp.results[last].frames.length - 1);
    }
    this.loop(0);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    window.removeEventListener("keydown", this.onKey);
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    this.renderer.dispose();
    this.root.remove();
  }

  // ------------------------------------------------------------ units and formatting
  private fmt(v: number, d = 1) {
    if (!Number.isFinite(v)) return "–";
    return (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  /** A value in the model's unit shown in the chosen unit (or as it is). */
  private q(v: number, q: ConvQ, d = 2) {
    const u = this.inp.units;
    const from = u.model(q), to = u.show(q);
    const x = convertValue(v, q, from, to);
    // pick the number of decimals from the magnitude
    const a = Math.abs(x);
    const dd = a >= 1000 ? 0 : a >= 100 ? 1 : d;
    return `${this.fmt(x, dd)}${to ? ` ${to}` : ""}`;
  }
  private qv(v: number, q: ConvQ) {
    const u = this.inp.units;
    return convertValue(v, q, u.model(q), u.show(q));
  }
  private unit(q: ConvQ) {
    return this.inp.units.show(q) ?? "";
  }
  /** Section dimensions in mm (when the model has a unit system in m or mm). */
  private dim(v: number) {
    const m = this.inp.units.model("length");
    if (m === "m") return this.fmt(v * 1000, v * 1000 < 10 ? 1 : 0);
    return this.fmt(v, Math.abs(v) < 10 ? 1 : 0);
  }
  private dimUnit() {
    const m = this.inp.units.model("length");
    return m === "m" || m === "mm" ? " mm" : m ? ` ${m}` : "";
  }

  private partDesc(p: VisPart): string {
    const d = (v: number) => this.dim(v);
    const u = this.dimUnit();
    const v = p.dims;
    if (p.kind === "beam" || p.kind === "truss") {
      switch (p.shape) {
        case "I":
          return `I ${d(v[1])} × ${d(Math.max(v[2], v[3]))}${u}, flange ${d(v[4])}, web ${d(v[6])}`;
        case "RECT":
          return `Rectangle ${d(v[0])} × ${d(v[1])}${u}`;
        case "CIRC":
          return `Circle Ø ${d(2 * v[0])}${u}`;
        case "PIPE":
        case "THICK PIPE":
          return `Pipe Ø ${d(2 * v[0])} × ${d(v[1])}${u}`;
        case "BOX":
          return `Box ${d(v[0])} × ${d(v[1])} × ${d(v[2])}${u}`;
        case "L":
          return `L ${d(v[0])} × ${d(v[1])} × ${d(v[2])}${u}`;
        case "CHANNEL":
          return `Channel ${d(v[1])} × ${d(v[2])}${u}`;
        case "HEX":
          return `Hexagon ${d(v[0])} × ${d(v[1])}${u}`;
        case "TRAPEZOID":
          return `Trapezoid ${d(v[0])} / ${d(v[2])} × ${d(v[1])}${u}`;
        case null:
        case undefined:
          break;
        default:
          return `${p.shape} ${v.map(d).join(" × ")}${u}`;
      }
      if (p.area) {
        const m = this.inp.units.model("length");
        const a = m === "m" ? p.area * 1e6 : p.area;
        return `A = ${this.fmt(a, a < 100 ? 1 : 0)}${m === "m" || m === "mm" ? " mm²" : ""}`;
      }
      return p.kind === "truss" ? "Truss" : "Beam";
    }
    if (p.kind === "shell" || p.kind === "membrane") return `${p.kind === "shell" ? "Shell" : "Membrane"}, t = ${p.thickness ? d(p.thickness) + u : "?"}`;
    if (p.kind === "plane") return `Plane, t = ${p.thickness ? d(p.thickness) + u : "?"} · ${p.elements} elements`;
    if (p.kind === "solid") return `Solid · ${p.elements} elements`;
    return `${p.kind} · ${p.elements} elements`;
  }

  // ------------------------------------------------------------ the panel
  private buildUi() {
    const r = this.root;
    const panel = el("div", "vis-panel");
    const head = el("div", "vis-head");
    head.append(el("h1", "", this.inp.name), el("p", "vis-sub", ""));
    panel.appendChild(head);
    const rows = el("div", "vis-rows");
    panel.appendChild(rows);
    const res = el("div", "vis-opts");
    res.innerHTML = `<b>Results</b>
      <span class="vis-pr vis-none">No results. Solve the model (F5) to show the deformation, section forces and reactions.</span>
      <div class="vis-res" style="display:none">
        <label>Step <select class="vis-step"></select></label>
        <label class="vis-frame-row">Frame <select class="vis-frame"></select></label>
        <label><input type="checkbox" class="vis-def" checked> Show deformation</label>
        <label><input type="checkbox" class="vis-anim" checked> Animate</label>
        <label>Scale <input type="range" class="vis-scale" min="0" max="100" value="25"> <span class="vis-scaletxt"></span></label>
        <div class="vis-pr vis-max"></div>
      </div>`;
    panel.appendChild(res);
    const opts = el("div", "vis-opts");
    opts.innerHTML = `
      <label><input type="checkbox" class="vis-transp"> Transparent plates and solids</label>
      <label><input type="checkbox" class="vis-edges" checked> Edge lines</label>
      <label><input type="checkbox" class="vis-render"> Rendering mode (shadows, sky, ground)</label>
      <label>Colour <select class="vis-scheme">
        <option value="parts">Colour per part</option><option value="grey">Grey painted steel</option>
        <option value="green">Bridge green</option><option value="white">White</option>
        <option value="red">Red</option><option value="corten">Weathering steel</option>
      </select></label>
      <label>Sun <input type="range" class="vis-sun" min="0" max="360" value="55"></label>
      <div class="vis-views"><button type="button" class="vis-png">Save image (4K PNG)</button></div>
      <div class="vis-views">
        <button type="button" data-view="iso">Iso</button><button type="button" data-view="side">Side</button>
        <button type="button" data-view="top">Top</button><button type="button" data-view="end">End</button>
        <button type="button" data-view="under">Below</button><button type="button" data-view="fit">Fit</button>
      </div>`;
    panel.appendChild(opts);
    r.appendChild(panel);
    const info = el("div", "vis-info");
    const dock = el("div", "vis-dock");
    dock.innerHTML = `<div class="vis-dh"><span class="vis-dockhead"></span><span class="vis-x vis-dockclose" title="Close the cut (Esc)">✕</span></div>
      <div class="vis-axisrow"></div><div class="vis-dockvals"></div><div class="vis-plots"></div>
      <div class="vis-pr vis-docknote"></div>`;
    const hint = el("div", "vis-hint", "Drag = rotate · right-drag = pan · scroll = zoom · click a part for its section forces");
    const tips = el("div", "vis-tips");
    r.append(info, dock, hint, tips);
    const q = <T extends HTMLElement>(c: string, from: HTMLElement = r) => from.querySelector(`.${c}`) as T;
    this.ui = {
      panel, rows, sub: q("vis-sub"), info, dock, hint, tips,
      none: q("vis-none"), res: q("vis-res"), step: q("vis-step"), frame: q("vis-frame"), frameRow: q("vis-frame-row"),
      def: q("vis-def"), anim: q("vis-anim"), scale: q("vis-scale"), scaleTxt: q("vis-scaletxt"), max: q("vis-max"),
      transp: q("vis-transp"), edges: q("vis-edges"), render: q("vis-render"), scheme: q("vis-scheme"), sun: q("vis-sun"), png: q("vis-png"),
      dockHead: q("vis-dockhead"), dockVals: q("vis-dockvals"), plots: q("vis-plots"), dockClose: q("vis-dockclose"), axisRow: q("vis-axisrow"), dockNote: q("vis-docknote"),
    };
    const u = this.ui;
    const chk = (k: string) => (u[k] as HTMLInputElement).checked;
    u.def.onchange = () => (this.opt.def = chk("def"));
    u.anim.onchange = () => (this.opt.anim = chk("anim"));
    u.scale.oninput = () => {
      this.opt.scale = +(u.scale as HTMLInputElement).value;
      this.updateResUi();
    };
    u.transp.onchange = () => {
      this.opt.transp = chk("transp");
      this.applyOpts();
    };
    u.edges.onchange = () => {
      this.opt.edges = chk("edges");
      this.applyOpts();
    };
    u.render.onchange = () => {
      this.opt.render = chk("render");
      if (this.opt.render) {
        (u.edges as HTMLInputElement).checked = this.opt.edges = false;
        this.applyOpts();
      }
      this.applyRender();
    };
    u.scheme.onchange = () => {
      this.opt.scheme = (u.scheme as HTMLSelectElement).value;
      this.applyRender();
    };
    u.sun.oninput = () => {
      this.opt.sun = +(u.sun as HTMLInputElement).value;
      this.applyRender();
    };
    u.png.onclick = () => this.savePng();
    r.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((b) => (b.onclick = () => this.setView(b.dataset.view!)));
    u.dockClose.onclick = () => this.clearSel();
    if (this.hasResults) {
      u.none.style.display = "none";
      u.res.style.display = "";
      const st = u.step as HTMLSelectElement;
      this.inp.results.forEach((s, i) => st.add(new Option(s.name, String(i))));
      st.onchange = () => {
        const s = +st.value;
        this.setStep(s, this.inp.results[s].frames.length - 1);
      };
      (u.frame as HTMLSelectElement).onchange = () => this.setStep(this.step, +(u.frame as HTMLSelectElement).value);
    }
    // drag in a diagram = move the cut
    const box = u.plots;
    const scrub = (e: PointerEvent) => {
      if (!this.sel) return;
      const rr = box.getBoundingClientRect(), x = (e.clientX - rr.left) % (this.plotW + 14);
      const f = Math.min(1, Math.max(0, (x - 46) / (this.plotW - 46 - 12)));
      const [a, b] = this.plotRange();
      this.setCut(a + f * (b - a), true);
    };
    box.addEventListener("pointerdown", (e) => {
      if (this.sel) {
        box.setPointerCapture(e.pointerId);
        scrub(e);
      }
    });
    box.addEventListener("pointermove", (e) => {
      if (this.sel && e.buttons) scrub(e);
    });
  }

  private group(part: number, name: string, desc: string, color: number): Group {
    const g = new THREE.Group();
    this.scene.add(g);
    const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.25, roughness: 0.6, side: THREE.DoubleSide });
    const lineMat = new THREE.LineBasicMaterial({ color: this.inp.dark ? 0x0b0d10 : 0x26303b, transparent: true, opacity: 0.55 });
    const o: Group = { part, name, desc, color, g, mat, lineMat, recs: [], visible: true };
    this.groups.push(o);
    return o;
  }

  private addRec(o: Group, kind: Rec["kind"], geom: THREE.BufferGeometry, edges: THREE.BufferGeometry | null, maps: VertMap[]): Rec {
    const m = new THREE.Mesh(geom, o.mat);
    m.castShadow = m.receiveShadow = true;
    m.frustumCulled = false;
    o.g.add(m);
    let l: THREE.LineSegments | null = null;
    if (edges && edges.attributes.position.count) {
      l = new THREE.LineSegments(edges, o.lineMat);
      l.frustumCulled = false;
      o.g.add(l);
    }
    const attrs = [geom.attributes.position as THREE.BufferAttribute, ...(l ? [l.geometry.attributes.position as THREE.BufferAttribute] : [])];
    const rec: Rec = { kind, group: o, mesh: m, edges: l, map: maps.slice(0, attrs.length), base: attrs.map((a) => (a.array as Float32Array).slice()), disp: attrs.map(() => null), attrs };
    m.userData.rec = rec;
    o.recs.push(rec);
    return rec;
  }

  // ------------------------------------------------------------ geometry from the model
  private buildGeometry() {
    const { vm, mesh } = this.inp;
    const pos = mesh.positions;
    const partColor = (i: number) => PALETTE[i % PALETTE.length];
    const groups = vm.parts.map((p, i) => this.group(i, p.name, this.partDesc(p), partColor(i)));
    // members: the profile swept along the path
    for (let mi = 0; mi < vm.members.length; mi++) {
      const m = vm.members[mi];
      const part = vm.parts[m.part];
      const pts = m.nodes.map((n) => vec(pos, 3 * n));
      const t = pts.map((_, i) => pts[Math.min(pts.length - 1, i + 1)].clone().sub(pts[Math.max(0, i - 1)]).normalize());
      const path: MemberPath = { m, pts, t, n1: m.n1.map((a) => V3(a[0], a[1], a[2])), n2: m.n2.map((a) => V3(a[0], a[1], a[2])), beam: part.kind === "beam" };
      this.paths.push(path);
      const prof: VisProfile = vm.profiles[part.profile] ?? this.defaultProfile();
      const { geom, edges, maps } = this.sweep(path, prof);
      const rec = this.addRec(groups[m.part], "member", geom, edges, maps);
      rec.member = mi;
    }
    // shells, membranes, planar parts: slabs with their thickness; solids: their surface
    const te = mesh.triElem;
    const shellData = mesh.profiles?.shell;
    const byKey = new Map<string, number[]>();
    for (let k = 0; k < te.length; k++) {
      const e = te[k], p = vm.elemPart[e];
      if (p < 0) continue;
      const kind = vm.parts[p].kind;
      if (kind === "beam" || kind === "truss") continue;
      const key = kind === "solid" || kind === "plane" ? `${p}/${vm.elemBody[e]}` : `${p}/-1`;
      let l = byKey.get(key);
      if (!l) byKey.set(key, (l = []));
      l.push(k);
    }
    for (const [key, list] of byKey) {
      const [p, b] = key.split("/").map(Number);
      const part = vm.parts[p];
      const slab = part.kind === "shell" || part.kind === "membrane" || part.kind === "plane" || part.kind === "axisymmetric";
      const { geom, edges, maps } = slab ? this.slab(list, part, shellData) : this.surface(list);
      const rec = this.addRec(groups[p], b >= 0 ? "body" : part.kind === "solid" ? "part" : "shell", geom, edges, maps);
      if (b >= 0) rec.body = b;
    }
    // the panel rows
    const rows = this.ui.rows;
    let mass = 0;
    for (const o of this.groups) {
      if (!o.recs.length) continue;
      const p = this.inp.vm.parts[o.part];
      mass += p.mass;
      const row = el("label", "vis-row");
      row.innerHTML = `<input type="checkbox" checked><span class="vis-sw" style="background:#${o.color.toString(16).padStart(6, "0")}"></span>
        <span><span class="vis-nm">${o.name}</span><br><span class="vis-pr">${o.desc}</span></span>`;
      (row.querySelector("input") as HTMLInputElement).onchange = (e) => {
        o.visible = (e.target as HTMLInputElement).checked;
        o.g.visible = o.visible;
      };
      rows.appendChild(row);
    }
    const sys = this.inp.units.system;
    const kg = sys === "si" ? mass : sys === "si_mm" ? mass * 1000 : 0;
    const counts = [vm.members.length && `${vm.members.length} members`, vm.bodies.length && `${vm.bodies.length} bod${vm.bodies.length === 1 ? "y" : "ies"}`].filter(Boolean).join(" · ");
    this.ui.sub.textContent = [counts, kg > 0 ? `mass ≈ ${Math.round(kg).toLocaleString("en-US")} kg` : ""].filter(Boolean).join(" · ");
  }

  private defaultProfile(): VisProfile {
    const r = this.R * 0.006;
    return { outer: Array.from({ length: 16 }, (_, k) => [r * Math.cos((2 * Math.PI * k) / 16), r * Math.sin((2 * Math.PI * k) / 16)] as [number, number]), hole: null, pieces: [], smooth: true };
  }

  /** Sweeps a profile along a member path (subdivided so it can bend). */
  private sweep(path: MemberPath, prof: VisProfile) {
    const { pts, n1, n2 } = path;
    const n = pts.length;
    // rings: (segment, ξ, position, n1, n2)
    const rings: { seg: number; xi: number; p: THREE.Vector3; a: THREE.Vector3; b: THREE.Vector3 }[] = [];
    for (let i = 0; i < n - 1; i++) {
      const L = pts[i].distanceTo(pts[i + 1]);
      const k = path.beam ? Math.max(1, Math.min(12, Math.ceil(L / (this.R / 40)))) : 1;
      for (let j = i === 0 ? 0 : 1; j <= k; j++) {
        const xi = j / k;
        const a = n1[i].clone().lerp(n1[i + 1], xi).normalize(), b = n2[i].clone().lerp(n2[i + 1], xi).normalize();
        rings.push({ seg: i, xi, p: pts[i].clone().lerp(pts[i + 1], xi), a, b });
      }
    }
    const pos: number[] = [], seg: number[] = [], xis: number[] = [], nrm: (THREE.Vector3 | null)[] = [];
    const epos: number[] = [], eseg: number[] = [], exi: number[] = [];
    const at = (r: (typeof rings)[0], q: [number, number]) => r.p.clone().addScaledVector(r.a, q[0]).addScaledVector(r.b, q[1]);
    const push = (r: (typeof rings)[0], v: THREE.Vector3, nm: THREE.Vector3 | null) => {
      pos.push(v.x, v.y, v.z);
      seg.push(r.seg);
      xis.push(r.xi);
      nrm.push(nm);
    };
    const epush = (r: (typeof rings)[0], v: THREE.Vector3) => {
      epos.push(v.x, v.y, v.z);
      eseg.push(r.seg);
      exi.push(r.xi);
    };
    const loops: { loop: [number, number][]; smooth: boolean; hole: [number, number][] | null }[] = prof.outer.length
      ? [{ loop: prof.outer, smooth: prof.smooth, hole: prof.hole }]
      : prof.pieces.map((p) => ({ loop: p, smooth: false, hole: null }));
    for (const L of loops) {
      for (const loop of [L.loop, L.hole].filter(Boolean) as [number, number][][]) {
        const m = loop.length;
        for (let i = 0; i < rings.length - 1; i++)
          for (let j = 0; j < m; j++) {
            const k = (j + 1) % m, r0 = rings[i], r1 = rings[i + 1];
            const A = at(r0, loop[j]), B = at(r0, loop[k]), C = at(r1, loop[k]), D = at(r1, loop[j]);
            const rn = (r: (typeof rings)[0], q: [number, number]) => (L.smooth ? r.a.clone().multiplyScalar(q[0]).addScaledVector(r.b, q[1]).normalize() : null);
            push(r0, A, rn(r0, loop[j]));
            push(r0, B, rn(r0, loop[k]));
            push(r1, C, rn(r1, loop[k]));
            push(r0, A, rn(r0, loop[j]));
            push(r1, C, rn(r1, loop[k]));
            push(r1, D, rn(r1, loop[j]));
          }
        // edge lines along the member at the sharp corners of the profile
        if (!L.smooth)
          for (let j = 0; j < m; j++) {
            const a = loop[(j + m - 1) % m], b = loop[j], c = loop[(j + 1) % m];
            const d1 = V2(b[0] - a[0], b[1] - a[1]).normalize(), d2 = V2(c[0] - b[0], c[1] - b[1]).normalize();
            if (d1.dot(d2) > Math.cos((30 * Math.PI) / 180)) continue;
            for (let i = 0; i < rings.length - 1; i++) {
              epush(rings[i], at(rings[i], b));
              epush(rings[i + 1], at(rings[i + 1], b));
            }
          }
        // outline at the ends
        for (const r of [rings[0], rings[rings.length - 1]])
          for (let j = 0; j < m; j++) {
            epush(r, at(r, loop[j]));
            epush(r, at(r, loop[(j + 1) % m]));
          }
      }
      // end caps
      const faces = THREE.ShapeUtils.triangulateShape(L.loop.map((q) => V2(q[0], q[1])), L.hole ? [L.hole.map((q) => V2(q[0], q[1]))] : []);
      const all = L.loop.concat(L.hole ?? []);
      for (const r of [rings[0], rings[rings.length - 1]]) for (const [a, b, c] of faces) for (const v of [a, b, c]) push(r, at(r, all[v]), null);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geom.computeVertexNormals();
    const na = geom.attributes.normal as THREE.BufferAttribute;
    nrm.forEach((q, v) => {
      if (!q) return;
      const sg = na.getX(v) * q.x + na.getY(v) * q.y + na.getZ(v) * q.z < 0 ? -1 : 1;
      na.setXYZ(v, sg * q.x, sg * q.y, sg * q.z);
    });
    const edges = new THREE.BufferGeometry();
    edges.setAttribute("position", new THREE.Float32BufferAttribute(epos, 3));
    return { geom, edges, maps: [{ seg: Int32Array.from(seg), xi: Float32Array.from(xis) }, { seg: Int32Array.from(eseg), xi: Float32Array.from(exi) }] as VertMap[] };
  }

  /** Feature edges of a triangle set given by node triples: boundary edges and edges with a fold > 30°. */
  private featureEdges(tri: number[], P: (v: number) => THREE.Vector3) {
    const nrm: THREE.Vector3[] = [];
    for (let k = 0; k < tri.length; k += 3) {
      const a = P(tri[k]), b = P(tri[k + 1]), c = P(tri[k + 2]);
      nrm.push(b.sub(a).cross(c.sub(a)).normalize());
    }
    const m = new Map<string, [number, number, number[]]>();
    for (let t = 0; t < tri.length / 3; t++)
      for (let j = 0; j < 3; j++) {
        const a = tri[3 * t + j], b = tri[3 * t + ((j + 1) % 3)];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        const e = m.get(key);
        if (e) e[2].push(t);
        else m.set(key, [a, b, [t]]);
      }
    const out: number[] = [];
    const cos = Math.cos((30 * Math.PI) / 180);
    for (const [a, b, ts] of m.values()) {
      if (ts.length === 1 || (ts.length === 2 && Math.abs(nrm[ts[0]].dot(nrm[ts[1]])) < cos) || ts.length > 2) out.push(a, b);
    }
    return out;
  }

  /** The surface of solid elements (flat shading, vertices follow their nodes). */
  private surface(list: number[]) {
    const { tris, positions: pos } = this.inp.mesh;
    const tri: number[] = [];
    for (const k of list) tri.push(tris[3 * k], tris[3 * k + 1], tris[3 * k + 2]);
    const p: number[] = [];
    for (const n of tri) p.push(pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    geom.computeVertexNormals();
    const fe = this.featureEdges(tri, (v) => vec(pos, 3 * v));
    const ep: number[] = [];
    for (const n of fe) ep.push(pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]);
    const edges = new THREE.BufferGeometry();
    edges.setAttribute("position", new THREE.Float32BufferAttribute(ep, 3));
    return { geom, edges, maps: [{ node: Int32Array.from(tri) }, { node: Int32Array.from(fe) }] as VertMap[] };
  }

  /** Shells and planar elements as slabs: both faces offset by ±t/2 along the node normals, closed along the boundary. */
  private slab(list: number[], part: VisPart, shell: Float32Array | undefined) {
    const { tris, triElem, positions: pos } = this.inp.mesh;
    const planar = part.kind === "plane" || part.kind === "axisymmetric";
    // node normals and thickness (per node: the mean of its elements in the part)
    const nN = new Map<number, THREE.Vector3>(), tN = new Map<number, number>(), zN = new Map<number, number>(), cnt = new Map<number, number>();
    const ref = new Map<number, THREE.Vector3>();
    for (const k of list) {
      const e = triElem[k];
      let t = part.thickness ?? 0, z = 0;
      let nv: THREE.Vector3;
      if (!planar && shell && shell[5 * e] > 0) {
        t = shell[5 * e];
        z = shell[5 * e + 1];
        nv = V3(shell[5 * e + 2], shell[5 * e + 3], shell[5 * e + 4]);
      } else {
        const a = vec(pos, 3 * tris[3 * k]), b = vec(pos, 3 * tris[3 * k + 1]), c = vec(pos, 3 * tris[3 * k + 2]);
        nv = b.sub(a).cross(c.sub(a)).normalize();
      }
      if (planar) {
        nv = V3(0, 0, 1);
        t = Math.min(t || this.R * 0.02, this.R * 0.08);
      }
      for (let j = 0; j < 3; j++) {
        const n = tris[3 * k + j];
        const r = ref.get(n);
        const s = r && r.dot(nv) < 0 ? -1 : 1;
        if (!r) ref.set(n, nv.clone());
        nN.set(n, (nN.get(n) ?? V3(0, 0, 0)).addScaledVector(nv, s));
        tN.set(n, (tN.get(n) ?? 0) + t);
        zN.set(n, (zN.get(n) ?? 0) + s * z);
        cnt.set(n, (cnt.get(n) ?? 0) + 1);
      }
    }
    // vertex ids: 2·node + side (0 = −t/2, 1 = +t/2)
    const P = (v: number) => {
      const n = v >> 1, c = cnt.get(n) ?? 1, nv = (nN.get(n) ?? V3(0, 0, 1)).clone().normalize();
      const off = (zN.get(n) ?? 0) / c + ((v & 1 ? 0.5 : -0.5) * (tN.get(n) ?? 0)) / c;
      return vec(pos, 3 * n).addScaledVector(nv, off);
    };
    const tri: number[] = [];
    for (const k of list) {
      const [a, b, c] = [tris[3 * k], tris[3 * k + 1], tris[3 * k + 2]];
      tri.push(2 * a + 1, 2 * b + 1, 2 * c + 1, 2 * a, 2 * c, 2 * b);
    }
    // side walls along the boundary of the part
    const em = new Map<string, number>();
    for (const k of list)
      for (let j = 0; j < 3; j++) {
        const a = tris[3 * k + j], b = tris[3 * k + ((j + 1) % 3)];
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        em.set(key, (em.get(key) ?? 0) + 1);
      }
    for (const k of list)
      for (let j = 0; j < 3; j++) {
        const a = tris[3 * k + j], b = tris[3 * k + ((j + 1) % 3)];
        if (em.get(a < b ? `${a},${b}` : `${b},${a}`) !== 1) continue;
        tri.push(2 * a, 2 * b, 2 * b + 1, 2 * a, 2 * b + 1, 2 * a + 1);
      }
    const p: number[] = [];
    for (const v of tri) {
      const q = P(v);
      p.push(q.x, q.y, q.z);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    geom.computeVertexNormals();
    const fe = this.featureEdges(tri, P);
    const ep: number[] = [];
    for (const v of fe) {
      const q = P(v);
      ep.push(q.x, q.y, q.z);
    }
    const edges = new THREE.BufferGeometry();
    edges.setAttribute("position", new THREE.Float32BufferAttribute(ep, 3));
    return { geom, edges, maps: [{ node: Int32Array.from(tri, (v) => v >> 1) }, { node: Int32Array.from(fe, (v) => v >> 1) }] as VertMap[] };
  }

  // ------------------------------------------------------------ supports
  private buildSupportsFromSymbols() {
    // without results: the boundary conditions of the first step, grouped by their constraints
    if (this.hasResults) return;
    this.inp.engine
      .symbols(this.inp.id, this.inp.results.length ? 0 : null)
      .then((sy) => {
        if (this.disposed) return;
        const by = new Map<number, number[]>();
        sy.bcNodes.forEach((n, i) => {
          const m = sy.bcMasks[i] & 0x3f;
          if (!m) return;
          let l = by.get(m);
          if (!l) by.set(m, (l = []));
          l.push(n);
        });
        const pos = this.inp.mesh.positions;
        const list = [...by].map(([mask, nodes]) => {
          const c = nodes.reduce((a, n) => a.add(vec(pos, 3 * n)), V3(0, 0, 0)).multiplyScalar(1 / nodes.length);
          return { nodes, mask, center: [c.x, c.y, c.z] as Vec3, f: [0, 0, 0] as Vec3, m: [0, 0, 0] as Vec3 };
        });
        this.buildSupports(list);
      })
      .catch(() => {});
  }

  private buildSupports(list: VisFrame["supports"]) {
    if (this.supGroup) {
      this.scene.remove(this.supGroup.g);
      this.groups.splice(this.groups.indexOf(this.supGroup), 1);
      this.ui.rows.querySelector(".vis-suprow")?.remove();
    }
    this.supports = [];
    if (!list.length) return;
    const o = this.group(-1, "Supports", "", SUPPORT);
    o.isSupport = true;
    this.supGroup = o;
    const pos = this.inp.mesh.positions;
    const h = this.R * 0.045;
    list.forEach((s, si) => {
      const nodes = s.nodes.length > 60 ? s.nodes.filter((_, i) => i % Math.ceil(s.nodes.length / 60) === 0) : s.nodes;
      const size = s.nodes.length === 1 ? h : h * 0.45;
      const trans = s.mask & 7, rot = s.mask & 56;
      const transAll = this.is2d ? (trans & 3) === 3 : trans === 7;
      const obj: SupportObj = { index: si, open: false, recs: [], g: null, tips: [] };
      for (const n of nodes) {
        const P = vec(pos, 3 * n);
        let geo: THREE.BufferGeometry;
        if (transAll && rot) {
          geo = new THREE.BoxGeometry(size * 1.1, size * 0.5, this.is2d ? size * 0.3 : size * 1.1).translate(P.x, P.y - size * 0.25, P.z);
        } else {
          geo = new THREE.ConeGeometry(size * 0.6, size, 4).rotateY(Math.PI / 4).translate(P.x, P.y - size / 2, P.z);
        }
        if (geo.index) geo = geo.toNonIndexed();
        const node = new Int32Array(geo.attributes.position.count).fill(n);
        const rec = this.addRec(o, "support", geo, null, [{ node }]);
        rec.support = si;
        obj.recs.push(rec);
        if (!transAll && !(trans === 0)) {
          for (const dx of [-0.3, 0.3]) {
            const r = new THREE.CylinderGeometry(size * 0.1, size * 0.1, size, 16).rotateX(Math.PI / 2).translate(P.x + dx * size, P.y - size * 1.12, P.z).toNonIndexed();
            const rr = this.addRec(o, "support", r, null, [{ node: new Int32Array(r.attributes.position.count).fill(n) }]);
            rr.support = si;
            obj.recs.push(rr);
          }
        }
      }
      this.supports.push(obj);
    });
    const kinds = list.map((s) => this.supportKind(s.mask, s.nodes.length));
    o.desc = `${list.length} support${list.length === 1 ? "" : "s"} · ${[...new Set(kinds)].join(", ")}`;
    const row = el("label", "vis-row vis-suprow");
    row.innerHTML = `<input type="checkbox" ${o.visible ? "checked" : ""}><span class="vis-sw" style="background:#${SUPPORT.toString(16).padStart(6, "0")}"></span>
      <span><span class="vis-nm">${o.name}</span><br><span class="vis-pr">${o.desc}${this.hasResults ? " · click one for its reactions" : ""}</span></span>`;
    (row.querySelector("input") as HTMLInputElement).onchange = (e) => {
      o.visible = (e.target as HTMLInputElement).checked;
      this.applyRender();
    };
    this.ui.rows.appendChild(row);
    this.applyOpts();
    this.applyRender();
    this.lastK = null;
  }

  private supportKind(mask: number, nodes = 1) {
    const t = mask & 7, r = mask & 56;
    const nt = [1, 2, 4].filter((b) => t & b).length, nr = [8, 16, 32].filter((b) => r & b).length;
    const full = this.is2d ? nt >= 2 : nt === 3;
    if (full && nodes > 2) return "fixed";
    if (full && (this.is2d ? r & 32 : nr === 3)) return "fixed";
    if (full && !r) return "pinned";
    if (nt === 1 && !r) return "roller";
    return "guided";
  }

  // ------------------------------------------------------------ results
  private async setStep(step: number, frame: number) {
    this.step = step;
    this.frame = frame;
    const st = this.inp.results[step];
    const fs = this.ui.frame as HTMLSelectElement;
    fs.innerHTML = "";
    st.frames.forEach((f, i) => fs.add(new Option(f.description || `${(f.value ?? 0).toPrecision(4)}`, String(i))));
    fs.value = String(frame);
    (this.ui.step as HTMLSelectElement).value = String(step);
    this.ui.frameRow.style.display = st.frames.length > 1 ? "" : "none";
    try {
      const [U, F] = await Promise.all([this.inp.engine.displacement(this.inp.id, step, frame), this.inp.engine.visualFrame(this.inp.id, step, frame)]);
      if (this.disposed || this.step !== step || this.frame !== frame) return;
      this.U = U;
      this.F = F;
    } catch (e) {
      this.ui.max.textContent = String(e);
      return;
    }
    let mu = 0, mn = -1;
    for (let i = 0; i < this.U.length; i += 3) {
      const u = Math.hypot(this.U[i], this.U[i + 1], this.U[i + 2]);
      if (u > mu) {
        mu = u;
        mn = i / 3;
      }
    }
    this.maxU = { u: mu, node: mn };
    this.autoScale = mu > 0 ? (0.08 * this.R) / mu : 1;
    // round the automatic factor to 1, 2 or 5 × 10^n
    if (mu > 0) {
      const p = Math.pow(10, Math.floor(Math.log10(this.autoScale)));
      this.autoScale = [1, 2, 5, 10].map((m) => m * p).reduce((a, b) => (Math.abs(Math.log(b / this.autoScale)) < Math.abs(Math.log(a / this.autoScale)) ? b : a));
    }
    this.buildSupports(this.F.supports);
    this.computeDisp();
    this.updateResUi();
    this.refreshSelection();
  }

  private factor() {
    return (this.autoScale * this.opt.scale) / 25;
  }

  private updateResUi() {
    const f = this.factor();
    this.ui.scaleTxt.textContent = `× ${f >= 10 ? Math.round(f).toLocaleString("en-US") : this.fmt(f, f >= 1 ? 1 : 3)}`;
    if (this.maxU.node >= 0 && this.U) {
      const n = this.maxU.node, U = this.U, p = this.inp.mesh.positions;
      this.ui.max.innerHTML = `Largest displacement ${this.q(this.maxU.u, "length")} (vertical ${this.q(U[3 * n + 1], "length")}) at (${this.fmt(p[3 * n], 2)}, ${this.fmt(p[3 * n + 1], 2)}, ${this.fmt(p[3 * n + 2], 2)})`;
    } else this.ui.max.textContent = "";
  }

  /** Displacement of a member path point (segment i, ξ): Hermite with the rotations for beams. */
  private pathDisp(path: MemberPath, i: number, xi: number, out: THREE.Vector3) {
    const U = this.U!, rot = this.F?.rot;
    const a = path.m.nodes[i], b = path.m.nodes[Math.min(i + 1, path.m.nodes.length - 1)];
    const ua = vec(U, 3 * a), ub = vec(U, 3 * b);
    if (!path.beam || !rot || !rot.length || a === b) return out.copy(ua).lerp(ub, xi);
    const pa = path.pts[i], pb = path.pts[i + 1], L = pa.distanceTo(pb);
    const t = pb.clone().sub(pa).divideScalar(L || 1);
    const ta = vec(rot, 3 * a).cross(t), tb = vec(rot, 3 * b).cross(t);
    const [h1, h2, h3, h4] = hermite(xi);
    const axial = ua.dot(t) * (1 - xi) + ub.dot(t) * xi;
    const pa_ = ua.clone().addScaledVector(t, -ua.dot(t)), pb_ = ub.clone().addScaledVector(t, -ub.dot(t));
    const w = pa_.multiplyScalar(h1).addScaledVector(pb_, h3).addScaledVector(ta, L * h2).addScaledVector(tb, L * h4);
    w.addScaledVector(t, -w.dot(t));
    return out.copy(w).addScaledVector(t, axial);
  }

  private computeDisp() {
    if (!this.U) return;
    const U = this.U, tmp = V3(0, 0, 0);
    for (const o of this.groups)
      for (const rec of o.recs)
        rec.map.forEach((mp, k) => {
          const n = rec.base[k].length / 3;
          const d = (rec.disp[k] = rec.disp[k] ?? new Float32Array(3 * n));
          if ("node" in mp) {
            for (let v = 0; v < n; v++) {
              const nd = mp.node[v];
              d[3 * v] = U[3 * nd];
              d[3 * v + 1] = U[3 * nd + 1];
              d[3 * v + 2] = U[3 * nd + 2];
            }
          } else {
            const path = this.paths[rec.member!];
            for (let v = 0; v < n; v++) {
              this.pathDisp(path, mp.seg[v], mp.xi[v], tmp);
              d[3 * v] = tmp.x;
              d[3 * v + 1] = tmp.y;
              d[3 * v + 2] = tmp.z;
            }
          }
        });
    this.lastK = null;
  }

  private applyDeform(k: number) {
    if (k === this.lastK) return;
    this.lastK = k;
    for (const o of this.groups)
      for (const rec of o.recs)
        rec.attrs.forEach((attr, i) => {
          const a = attr.array as Float32Array, b = rec.base[i], d = rec.disp[i];
          if (!d) return;
          for (let j = 0; j < a.length; j++) a[j] = b[j] + k * d[j];
          attr.needsUpdate = true;
        });
  }

  private curK() {
    return this.lastK ?? 0;
  }

  // ------------------------------------------------------------ appearance
  private applyOpts() {
    this.setFocus();
    for (const o of this.groups) for (const r of o.recs) if (r.edges) r.edges.visible = this.opt.edges && !r.edges.userData.off;
  }

  private skyDome(radius: number) {
    const top = new THREE.Color(0x2f6fbd), hor = new THREE.Color(0xc9dcee), gr = new THREE.Color(0x7f8a6e);
    const g = new THREE.SphereGeometry(radius, 48, 24), col: number[] = [], pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const h = pos.getY(i) / radius;
      const c = h >= 0 ? hor.clone().lerp(top, Math.pow(h, 0.55)) : hor.clone().lerp(gr, Math.min(1, -h * 6));
      col.push(c.r, c.g, c.b);
    }
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false }));
  }

  private applyRender() {
    const on = this.opt.render, a = (this.opt.sun * Math.PI) / 180, R = this.R, C = this.center;
    this.sky.visible = this.ground.visible = on;
    if (this.supGroup) this.supGroup.g.visible = !on && this.supGroup.visible;
    this.scene.fog = on ? new THREE.Fog(0xc9dcee, 7 * R, 16 * R) : null;
    this.scene.environment = on ? this.envTex : null;
    this.renderer.toneMapping = on ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    this.renderer.toneMappingExposure = 0.85;
    this.sun.castShadow = on;
    this.sun.intensity = on ? 5.2 : 2.3;
    this.hemi.intensity = on ? 0.6 : 2.9;
    this.fill.intensity = on ? 0 : 0.9;
    this.sun.color.set(on ? 0xfff1dc : 0xffffff);
    if (on) this.sun.position.set(C.x + 2.2 * R * Math.cos(a), C.y + 2 * R, C.z + 2.2 * R * Math.sin(a));
    else this.sun.position.set(C.x + 1.3 * R, C.y + 2 * R, C.z + 1.6 * R);
    this.fill.position.set(C.x - R, C.y - 0.7 * R, C.z - 1.3 * R);
    for (const o of this.groups) {
      const c = this.opt.scheme === "parts" || o.isSupport ? o.color : SCHEMES[this.opt.scheme];
      o.mat.color.set(c);
      o.mat.metalness = on ? 0.35 : 0.25;
      o.mat.roughness = on ? (this.opt.scheme === "corten" ? 0.85 : 0.42) : 0.6;
      o.mat.envMapIntensity = 0.55;
      o.mat.needsUpdate = true;
    }
  }

  private setFocus() {
    // dims everything except the cut member or body
    const sel = this.sel;
    for (const o of this.groups) {
      const mine = sel && sel.recs.some((r) => r.group === o);
      const dim = sel && !mine;
      const plate = this.inp.vm.parts[o.part]?.kind;
      const tr = this.opt.transp && !o.isSupport && plate !== "beam" && plate !== "truss";
      const op = dim ? 0.07 : tr ? 0.35 : 1;
      o.mat.transparent = op < 1;
      o.mat.opacity = op;
      o.mat.depthWrite = op === 1;
      o.mat.needsUpdate = true;
      o.lineMat.opacity = dim ? 0.05 : 0.55;
    }
  }

  private async savePng() {
    const root = this.root, w = 3840, h = Math.round((w * root.clientHeight) / Math.max(1, root.clientWidth));
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
    const url = this.renderer.domElement.toDataURL("image/png");
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.resize();
    await this.inp.savePng(url);
  }

  // ------------------------------------------------------------ camera
  private setView(k: string) {
    const C = this.center;
    const views: Record<string, THREE.Vector3> = {
      iso: V3(-0.55, 0.4, 0.75),
      side: V3(0, 0.02, 1),
      top: V3(0, 1, 0.0005),
      end: V3(-1, 0.06, 0),
      under: V3(-0.5, -0.45, 0.6),
      fit: this.camera.position.clone().sub(this.controls.target),
    };
    const dir = (views[k] ?? views.iso).clone().normalize();
    // camera basis
    const fwd = dir.clone().negate();
    let right = fwd.clone().cross(V3(0, 1, 0));
    if (right.length() < 1e-6) right = V3(1, 0, 0);
    right.normalize();
    const up = right.clone().cross(fwd).normalize();
    const w = this.root.clientWidth || 1, h = this.root.clientHeight || 1;
    const panel = w > 640 ? Math.min(336, w * 0.4) : 0;
    const xc = panel / w, hw = 1 - xc; // the free area in NDC: centre and half width
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360), tanH = tanV * (w / h);
    let d = 0;
    const b = this.box;
    for (let i = 0; i < 8; i++) {
      const p = V3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z).sub(C);
      const z = p.dot(dir), x = Math.abs(p.dot(right)), y = Math.abs(p.dot(up));
      d = Math.max(d, z + x / (tanH * hw * 0.92), z + y / (tanV * 0.88));
    }
    d = Math.max(d, this.R * 0.5);
    // shift so the model sits in the middle of the free area
    const tgt = C.clone().addScaledVector(right, -xc * tanH * d);
    this.camera.position.copy(tgt).addScaledVector(dir, d);
    this.controls.target.copy(tgt);
    this.controls.update();
  }

  private resize() {
    const w = this.root.clientWidth, h = this.root.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.sel) this.renderDock();
  }

  private loop = (now: number) => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    if (this.U) {
      const sc = this.factor();
      this.applyDeform(!this.opt.def ? 0 : this.opt.anim ? sc * (0.5 - 0.5 * Math.cos((2 * Math.PI * (now || 0)) / 3000)) : sc);
    }
    this.updateCut();
    this.updateReactions();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  // ------------------------------------------------------------ members: positions along the path
  private frameAt(path: MemberPath, v: number) {
    const s = path.m.s;
    let i = 0;
    while (i < s.length - 2 && s[i + 1] < v) i++;
    const w = Math.min(1, Math.max(0, (v - s[i]) / (s[i + 1] - s[i] || 1)));
    const P = path.pts[i].clone().lerp(path.pts[i + 1], w);
    const t = path.pts[i + 1].clone().sub(path.pts[i]).normalize();
    const n1 = path.n1[i].clone().lerp(path.n1[i + 1], w).normalize(), n2 = path.n2[i].clone().lerp(path.n2[i + 1], w).normalize();
    return { P, t, n1, n2, i, w };
  }

  private deformedOnPath(path: MemberPath, v: number) {
    const fr = this.frameAt(path, v);
    if (!this.U) return fr.P;
    const d = this.pathDisp(path, fr.i, fr.w, V3(0, 0, 0));
    return fr.P.clone().addScaledVector(d, this.curK());
  }

  /** Section forces of a member in its local axes (N, Vy, Vz, T, My, Mz) at the stored points. */
  private memberForces(mi: number) {
    const F = this.F!.members[mi], path = this.paths[mi];
    const out: number[][] = [];
    for (let k = 0; k < F.s.length; k++) {
      const t = this.frameAt(path, F.s[k]).t, A = localAxes(t), f = F.f[k];
      const Fv = V3(f[0], f[1], f[2]), Mv = V3(f[3], f[4], f[5]);
      out.push([Fv.dot(A.x), Fv.dot(A.y), Fv.dot(A.z), Mv.dot(A.x), Mv.dot(A.y), Mv.dot(A.z)]);
    }
    return { s: F.s, f: out };
  }

  /** Section forces in local axes at the stations of the current selection. */
  private selForces(): { s: number[]; f: number[][] } | null {
    const r = this.selForcesRaw();
    if (!r) return null;
    // round-off (e.g. 1e-13 of the shear force) is shown as zero, not scaled up in a diagram
    let fm = 0, mm = 0;
    for (const f of r.f) {
      for (const c of [0, 1, 2]) fm = Math.max(fm, Math.abs(f[c]));
      for (const c of [3, 4, 5]) mm = Math.max(mm, Math.abs(f[c]));
    }
    const tol = [1, 1, 1, 0, 0, 0].map((k) => (k ? fm : mm) * 1e-7);
    return { s: r.s, f: r.f.map((f) => f.map((v, c) => (Math.abs(v) < tol[c] ? 0 : v))) };
  }

  private selForcesRaw(): { s: number[]; f: number[][] } | null {
    const sel = this.sel;
    if (!sel) return null;
    if (sel.kind === "member") return this.memberForces(sel.index);
    const d = sel.diag;
    if (!d) return null;
    const A = localAxes(sel.axis!);
    return {
      s: d.s,
      f: d.f.map((f) => {
        const Fv = V3(f[0], f[1], f[2]), Mv = V3(f[3], f[4], f[5]);
        return [Fv.dot(A.x), Fv.dot(A.y), Fv.dot(A.z), Mv.dot(A.x), Mv.dot(A.y), Mv.dot(A.z)];
      }),
    };
  }

  private forcesAt(v: number): number[] {
    const sf = this.selForces();
    const out = [0, 0, 0, 0, 0, 0];
    if (!sf || !sf.s.length) return out;
    const { s, f } = sf, n = s.length;
    let i = 0;
    while (i < n - 2 && s[i + 1] < v) i++;
    const w = n < 2 ? 0 : Math.min(1, Math.max(0, (v - s[i]) / (s[i + 1] - s[i] || 1)));
    const b = Math.min(n - 1, i + 1);
    for (let c = 0; c < 6; c++) out[c] = f[i][c] * (1 - w) + f[b][c] * w;
    return out;
  }

  private maxForces() {
    const sf = this.selForces(), mx = [0, 0, 0, 0, 0, 0];
    if (sf) for (const f of sf.f) for (let c = 0; c < 6; c++) mx[c] = Math.max(mx[c], Math.abs(f[c]));
    return mx;
  }

  // ------------------------------------------------------------ the cut
  private clearSel() {
    const sel = this.sel;
    if (!sel) return;
    sel.recs.forEach((r, i) => {
      r.mesh.material = sel.old[i][0];
      if (r.edges && sel.old[i][1]) r.edges.material = sel.old[i][1]!;
    });
    this.cutGroup.clear();
    this.sel = null;
    this.ui.tips.innerHTML = "";
    this.ui.dock.style.display = "none";
    this.ui.hint.style.display = "";
    this.setFocus();
  }

  private beginSel(kind: Sel["kind"], index: number, rec: Rec, recs: Rec[], vmin: number, vmax: number, v: number) {
    const plane = new THREE.Plane();
    const old: Sel["old"] = recs.map((r) => [r.mesh.material as THREE.Material, r.edges ? (r.edges.material as THREE.Material) : null]);
    for (const r of recs) {
      const mat = (r.mesh.material as THREE.MeshStandardMaterial).clone();
      mat.clippingPlanes = [plane];
      mat.transparent = false;
      mat.opacity = 1;
      mat.depthWrite = true;
      r.mesh.material = mat;
      if (r.edges) {
        const lm = (r.edges.material as THREE.LineBasicMaterial).clone();
        lm.clippingPlanes = [plane];
        lm.opacity = 0.55;
        r.edges.material = lm;
      }
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }));
    disc.matrixAutoUpdate = false;
    const arrows = new THREE.Group();
    const cap = new THREE.Group();
    cap.matrixAutoUpdate = false;
    this.cutGroup.add(disc, cap, arrows);
    this.sel = { kind, index, rec, recs, v, vmin, vmax, plane, old, cap, disc, arrows, tips: [], dirty: true };
    this.ui.dock.style.display = "block";
    this.ui.hint.style.display = "none";
    this.setFocus();
  }

  private selectMember(rec: Rec, v: number, move: boolean) {
    const mi = rec.member!;
    if (this.sel && (this.sel.kind !== "member" || this.sel.index !== mi)) this.clearSel();
    this.pick = null;
    this.ui.info.style.display = "none";
    if (!this.sel) {
      const path = this.paths[mi];
      this.beginSel("member", mi, rec, [rec], 0, path.m.length, v);
      // the cut face: the profile, light grey with a dark outline
      const part = this.inp.vm.parts[path.m.part];
      const prof = this.inp.vm.profiles[part.profile] ?? this.defaultProfile();
      const loops = prof.outer.length ? [{ o: prof.outer, h: prof.hole }] : prof.pieces.map((p) => ({ o: p, h: null }));
      const cap = this.sel!.cap as THREE.Group;
      for (const L of loops) {
        const shape = new THREE.Shape(L.o.map((q) => V2(q[0], q[1])));
        if (L.h) shape.holes.push(new THREE.Path(L.h.map((q) => V2(q[0], q[1]))));
        cap.add(new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: FACE, side: THREE.DoubleSide })));
        for (const loop of [L.o, L.h].filter(Boolean) as [number, number][][])
          cap.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(loop.map((q) => V3(q[0], q[1], 0.001 * this.R * 0.01))), new THREE.LineBasicMaterial({ color: FACE_EDGE })));
      }
      const size = this.profileSize(prof);
      this.sel!.disc.scale.setScalar(Math.max(size * 1.6, this.R * 0.03));
    }
    this.setCut(v);
    if (move) this.flyToCut();
  }

  private profileSize(prof: VisProfile) {
    let r = 0;
    for (const q of prof.outer.concat(...prof.pieces)) r = Math.max(r, Math.hypot(q[0], q[1]));
    return r || this.R * 0.01;
  }

  private bodyAxis(b: number, mode: Sel["axisMode"]) {
    const body = this.inp.vm.bodies[b];
    if (mode === "x") return V3(1, 0, 0);
    if (mode === "y") return V3(0, 1, 0);
    if (mode === "z") return V3(0, 0, 1);
    return V3(body.axis[0], body.axis[1], body.axis[2]);
  }

  private async selectBody(rec: Rec, p: THREE.Vector3, move: boolean, mode?: Sel["axisMode"]) {
    const b = rec.body!;
    const keepMode = this.sel && this.sel.kind === "body" && this.sel.index === b ? this.sel.axisMode : undefined;
    if (this.sel && (this.sel.kind !== "body" || this.sel.index !== b || mode)) this.clearSel();
    this.pick = null;
    this.ui.info.style.display = "none";
    const body = this.inp.vm.bodies[b];
    if (!this.sel) {
      const axisMode = mode ?? keepMode ?? "auto";
      const axis = this.bodyAxis(b, axisMode);
      const recs = this.groups.flatMap((o) => o.recs.filter((r) => r.kind === "body" && r.body === b));
      const C = V3(...body.center);
      const v = p.clone().sub(C).dot(axis);
      this.beginSel("body", b, rec, recs, -Infinity, Infinity, v);
      const sel = this.sel as unknown as Sel;
      sel.axis = axis;
      sel.axisMode = axisMode;
      sel.diag = null;
      sel.disc.scale.setScalar(Math.max(this.R * 0.25, 1e-9));
      sel.disc.visible = false;
      this.renderDock();
      try {
        const d = await this.inp.engine.visualDiagram(this.inp.id, this.step, this.frame, b, [axis.x, axis.y, axis.z], 101);
        if ((this.sel as Sel | null) !== sel) return;
        sel.diag = d;
        sel.vmin = d.s0;
        sel.vmax = d.s1;
      } catch (e) {
        this.ui.dockNote.textContent = String(e);
      }
      this.setCut(sel.v);
    } else this.setCut(p.clone().sub(V3(...body.center)).dot(this.sel.axis!));
    if (move) this.flyToCut();
  }

  private requestCut() {
    const sel = this.sel;
    if (!sel || sel.kind !== "body") return;
    if (sel.pending) {
      sel.want = sel.v;
      return;
    }
    sel.pending = true;
    const v = sel.v, ax = sel.axis!;
    this.inp.engine
      .visualCut(this.inp.id, this.step, this.frame, sel.index, [ax.x, ax.y, ax.z], v)
      .then((c) => {
        if (this.sel !== sel) return;
        sel.cut = c;
        this.buildBodyCap(c);
        sel.dirty = true;
      })
      .catch(() => {})
      .finally(() => {
        sel.pending = false;
        if (this.sel === sel && sel.want != null && sel.want !== v) {
          sel.want = null;
          this.requestCut();
        }
      });
  }

  private buildBodyCap(c: VisCut) {
    const sel = this.sel!;
    const cap = sel.cap as THREE.Group;
    cap.clear();
    // the cap is built in world coordinates relative to the centroid (moved with the deformation)
    const C = V3(...c.centroid);
    if (c.tris.length) {
      const p = new Float32Array(c.tris.length);
      for (let i = 0; i < c.tris.length; i += 3) {
        p[i] = c.tris[i] - C.x;
        p[i + 1] = c.tris[i + 1] - C.y;
        p[i + 2] = c.tris[i + 2] - C.z;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(p, 3));
      cap.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: FACE, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 })));
      // outline: the triangle edges that are not shared
      const key = (i: number) => `${p[3 * i].toFixed(6)},${p[3 * i + 1].toFixed(6)},${p[3 * i + 2].toFixed(6)}`;
      const cnt = new Map<string, [number, number, number]>();
      for (let t = 0; t < p.length / 9; t++)
        for (let j = 0; j < 3; j++) {
          const a = 3 * t + j, b = 3 * t + ((j + 1) % 3);
          const ka = key(a), kb = key(b), k = ka < kb ? ka + "|" + kb : kb + "|" + ka;
          const e = cnt.get(k);
          if (e) e[2]++;
          else cnt.set(k, [a, b, 1]);
        }
      const lp: number[] = [];
      for (const [a, b, n] of cnt.values()) if (n === 1) lp.push(p[3 * a], p[3 * a + 1], p[3 * a + 2], p[3 * b], p[3 * b + 1], p[3 * b + 2]);
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
      cap.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: FACE_EDGE })));
    }
    if (c.segs.length) {
      const lp = c.segs.map((x, i) => x - [C.x, C.y, C.z][i % 3]);
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
      cap.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: FORCE, linewidth: 2 })));
    }
  }

  private nearestNode(p: THREE.Vector3, body: number): number {
    // the body node closest to p (for moving the cut with the deformation)
    const { vm, mesh } = this.inp;
    let best = -1, bd = Infinity;
    const tris = mesh.tris, te = mesh.triElem, pos = mesh.positions;
    for (let k = 0; k < te.length; k++) {
      if (vm.elemBody[te[k]] !== body) continue;
      for (let j = 0; j < 3; j++) {
        const n = tris[3 * k + j];
        const d = (pos[3 * n] - p.x) ** 2 + (pos[3 * n + 1] - p.y) ** 2 + (pos[3 * n + 2] - p.z) ** 2;
        if (d < bd) {
          bd = d;
          best = n;
        }
      }
    }
    return best;
  }

  private setCut(v: number, follow = false) {
    const sel = this.sel;
    if (!sel) return;
    const old = follow ? this.cutPoint() : null;
    sel.v = Math.min(sel.vmax, Math.max(sel.vmin, v));
    sel.dirty = true;
    if (sel.kind === "body") this.requestCut();
    this.renderDock();
    if (follow && old && !this.fly) {
      const d = this.cutPoint().sub(old);
      this.camera.position.add(d);
      this.controls.target.add(d);
    }
  }

  /** The undeformed point of the cut and the axes there. */
  private cutFrame(): { P: THREE.Vector3; t: THREE.Vector3; n1: THREE.Vector3; n2: THREE.Vector3; D: THREE.Vector3 } {
    const sel = this.sel!;
    if (sel.kind === "member") {
      const path = this.paths[sel.index], fr = this.frameAt(path, sel.v);
      const D = this.U ? this.pathDisp(path, fr.i, fr.w, V3(0, 0, 0)).multiplyScalar(this.curK()) : V3(0, 0, 0);
      return { P: fr.P, t: fr.t, n1: fr.n1, n2: fr.n2, D };
    }
    const body = this.inp.vm.bodies[sel.index], ax = sel.axis!;
    const P = sel.cut && Math.abs(sel.cut.s - sel.v) < 1e-12 ? V3(...sel.cut.centroid) : V3(...body.center).addScaledVector(ax, sel.v);
    let D = V3(0, 0, 0);
    if (this.U && sel.cut) {
      const key = `${sel.cut.s}`;
      if (sel.rec.mesh.userData.nodeKey !== key) {
        sel.rec.mesh.userData.nodeKey = key;
        sel.rec.mesh.userData.node = this.nearestNode(V3(...sel.cut.centroid), sel.index);
      }
      const n = sel.rec.mesh.userData.node as number;
      if (n >= 0) D = vec(this.U, 3 * n).multiplyScalar(this.curK());
    }
    const A = localAxes(ax);
    return { P, t: ax, n1: A.y, n2: A.z.clone().negate(), D };
  }

  private cutPoint() {
    const f = this.cutFrame();
    return f.P.clone().add(f.D);
  }

  private flyToCut() {
    const fr = this.cutFrame(), P = fr.P.clone().add(fr.D), A = localAxes(fr.t);
    let side = this.camera.position.clone().sub(P);
    side.addScaledVector(fr.t, -side.dot(fr.t));
    if (side.length() < 1e-9) side.copy(A.y);
    side.normalize();
    const dir = A.x.clone().multiplyScalar(0.8).addScaledVector(side, 0.5).addScaledVector(A.z, -0.3).normalize();
    const size = this.sel!.kind === "member" ? Math.max(this.profileSize(this.inp.vm.profiles[this.inp.vm.parts[this.paths[this.sel!.index].m.part].profile] ?? this.defaultProfile()) * 14, this.R * 0.25) : this.R * 2.2;
    const drop = V3(0, -0.15 * size, 0);
    this.fly = { t0: performance.now(), p0: this.camera.position.clone(), q0: this.controls.target.clone(), p1: P.clone().addScaledVector(dir, size).add(drop), q1: P.clone().add(drop) };
  }

  private arrowSize() {
    const sel = this.sel!;
    if (sel.kind === "member") {
      const p = this.inp.vm.profiles[this.inp.vm.parts[this.paths[sel.index].m.part].profile];
      return Math.max(p ? this.profileSize(p) * 2.2 : 0, this.R * 0.05);
    }
    return this.R * 0.18;
  }

  private updateCut() {
    if (this.fly) {
      const s = Math.min(1, (performance.now() - this.fly.t0) / 700), e = s * s * (3 - 2 * s);
      this.camera.position.lerpVectors(this.fly.p0, this.fly.p1, e);
      this.controls.target.lerpVectors(this.fly.q0, this.fly.q1, e);
      if (s >= 1) this.fly = null;
    }
    const sel = this.sel;
    if (!sel) return;
    const fr = this.cutFrame(), P = fr.P.clone().add(fr.D), A = localAxes(fr.t);
    // the part on the +x side is removed: the visible face is the positive section
    sel.plane.setFromNormalAndCoplanarPoint(A.x.clone().negate(), P);
    const L = this.arrowSize(), eps = L * 0.004;
    if (sel.kind === "member") {
      const M = new THREE.Matrix4().makeBasis(fr.n1, fr.n2, fr.t);
      sel.cap.matrix.copy(M).setPosition(P.clone().addScaledVector(A.x, eps));
      sel.disc.matrix.copy(M).scale(sel.disc.scale).setPosition(P.clone().addScaledVector(A.x, eps * 0.5));
    } else {
      sel.cap.matrix.identity().setPosition(P.clone().addScaledVector(A.x, eps));
    }
    sel.arrows.position.copy(P);
    if (sel.dirty) {
      sel.dirty = false;
      sel.arrows.clear();
      sel.tips = [];
      const f = this.forcesAt(sel.v), mx = this.maxForces(), at = (a: THREE.Vector3, s: number) => a.clone().multiplyScalar(s);
      const tip = (pos: THREE.Vector3, html: string, cls: string) => sel.tips.push({ pos, html, cls });
      const r = L * 0.018;
      const axes = A as unknown as Record<string, THREE.Vector3>;
      for (const k of ["x", "y", "z"]) {
        sel.arrows.add(arrow(at(axes[k], k === "x" ? 0 : -0.3 * L), at(axes[k], 0.62 * L), AXIS, r * 0.3));
        tip(at(axes[k], 0.67 * L), `<i>${k}</i>`, "ax");
      }
      sel.arrows.add(new THREE.Mesh(new THREE.SphereGeometry(r * 0.7, 12, 8), new THREE.MeshBasicMaterial({ color: 0x000000 })));
      // normal force along x: tension leaves the face, compression pushes on it
      const off = at(A.x, 0.03 * L), lenN = (0.5 + (0.8 * Math.abs(f[0])) / (mx[0] || 1)) * L, far = at(A.x, lenN + 0.05 * L);
      const fmax = Math.max(mx[0], mx[1], mx[2]) || 1, mmax = Math.max(mx[3], mx[4], mx[5]) || 1;
      if (Math.abs(f[0]) > 1e-3 * fmax) {
        sel.arrows.add(f[0] >= 0 ? arrow(off, far, FORCE, r * 1.2) : arrow(far, off, FORCE, r * 1.2));
        tip(far, `${sym("N")} = ${this.q(f[0], "force")}`, "f");
      }
      const vmax = Math.max(mx[1], mx[2]) || 1;
      (
        [
          ["y", 1],
          ["z", 2],
        ] as const
      ).forEach(([k, c]) => {
        if (Math.abs(f[c]) < 0.02 * vmax || Math.abs(f[c]) < 1e-3 * fmax) return;
        const end = at(axes[k], Math.sign(f[c]) * (0.28 + (0.4 * Math.abs(f[c])) / vmax) * L).add(off);
        sel.arrows.add(arrow(off, end, FORCE, r * 0.8));
        tip(end, `${sym("V", k)} = ${this.q(f[c], "force")}`, "f");
      });
      // bending moments: vectors with a double head a little in front of the face
      const front = at(A.x, 0.2 * L);
      const mb = Math.max(mx[4], mx[5]) || 1;
      (
        [
          ["y", 4],
          ["z", 5],
        ] as const
      ).forEach(([k, c]) => {
        if (Math.abs(f[c]) < 0.02 * mb || Math.abs(f[c]) < 1e-3 * mmax) return;
        const end = at(axes[k], Math.sign(f[c]) * (0.34 + (0.4 * Math.abs(f[c])) / mb) * L).add(front);
        sel.arrows.add(arrow(front, end, FORCE, r * 0.8, 2));
        tip(end, `${sym("M", k)} = ${this.q(f[c], "moment")}`, "f");
      });
      if (Math.abs(f[3]) > 0.02 * mmax && Math.abs(f[3]) > 1e-9) {
        const s0 = at(A.x, 0.25 * L), e = at(A.x, (0.25 + 0.3 * Math.sign(f[3])) * L + 0.25 * L * (f[3] < 0 ? 1 : 0));
        sel.arrows.add(arrow(f[3] > 0 ? s0 : e, f[3] > 0 ? at(A.x, 0.75 * L) : s0, FORCE, r * 0.6, 2));
        tip(at(A.x, 0.8 * L), `${sym("T")} = ${this.q(f[3], "moment")}`, "f");
      }
      this.ui.tips.innerHTML = sel.tips.map((t) => `<div class="vis-lbl vis-lbl-${t.cls}">${t.html}</div>`).join("");
    }
    const w = this.root.clientWidth, h = this.root.clientHeight;
    sel.tips.forEach((t, i) => {
      const l = this.ui.tips.children[i] as HTMLElement, q = t.pos.clone().add(P).project(this.camera);
      if (!l) return;
      l.style.display = q.z < 1 ? "block" : "none";
      l.style.left = ((q.x + 1) / 2) * w + 7 + "px";
      l.style.top = ((1 - q.y) / 2) * h - 11 + "px";
    });
  }

  // ------------------------------------------------------------ the diagrams
  private plotRange(): [number, number] {
    const sel = this.sel!;
    if (sel.kind === "member") return [0, this.paths[sel.index].m.length];
    return [sel.diag?.s0 ?? 0, sel.diag?.s1 ?? 1];
  }

  /** The coordinate shown on the diagram axis: distance along a member; the global coordinate along a body's global axis. */
  private shownCoord(v: number) {
    const sel = this.sel!;
    if (sel.kind === "member") return { v, name: "s" };
    const a = sel.axis!, c = this.inp.vm.bodies[sel.index].center;
    const k = [a.x, a.y, a.z].findIndex((x) => Math.abs(x) === 1);
    if (k >= 0) return { v: c[k] + v * Math.sign([a.x, a.y, a.z][k]), name: "xyz"[k] };
    return { v: v - (sel.diag?.s0 ?? 0), name: "s" };
  }

  private plotSVG(P: { c: number; name: string; s: string; q: ConvQ; flip?: boolean }, v: number) {
    const sf = this.selForces()!;
    const xs = sf.s, ys = sf.f.map((f) => this.qv(f[P.c], P.q));
    const w = this.plotW, h = this.plotH, m = { l: 46, r: 12, t: 26, b: 22 };
    const [x0, x1] = this.plotRange();
    const dmin = Math.min(...ys), dmax = Math.max(...ys);
    let lo = Math.min(0, dmin), hi = Math.max(0, dmax);
    if (hi - lo < 1e-12 * Math.max(1, Math.abs(hi))) {
      hi = 1;
      lo = -1;
    }
    const pad = 0.14 * (hi - lo);
    lo -= pad;
    hi += pad;
    const X = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (w - m.l - m.r);
    const Y = (y: number) => m.t + ((P.flip ? y - lo : hi - y) / (hi - lo)) * (h - m.t - m.b);
    const line = xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
    const area = xs.length ? `${line}L${X(xs[xs.length - 1]).toFixed(1)},${Y(0).toFixed(1)}L${X(xs[0]).toFixed(1)},${Y(0).toFixed(1)}Z` : "";
    const cur = this.qv(this.forcesAt(v)[P.c], P.q), imax = ys.indexOf(dmax), imin = ys.indexOf(dmin);
    const dec = (x: number) => (Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? 1 : 2);
    const ext = (i: number, above: boolean) => `<circle cx="${X(xs[i]).toFixed(1)}" cy="${Y(ys[i]).toFixed(1)}" r="2.2" class="pt"/>
      <text class="ex" x="${Math.min(w - m.r - 12, Math.max(m.l + 12, X(xs[i]))).toFixed(1)}" y="${(Y(ys[i]) + (above ? -6 : 14)).toFixed(1)}" text-anchor="middle">${this.fmt(ys[i], dec(ys[i]))}</text>`;
    const title = `${P.name} ${P.s}`.replace(/<i>/g, '<tspan class="it">').replace(/<\/i>/g, "</tspan>").replace(/<sub>/g, '<tspan class="sb" dy="3">').replace(/<\/sub>/g, "</tspan>");
    const unit = this.unit(P.q);
    const tx = niceTicks(x0, x1, 5);
    const c0 = this.shownCoord(x0).v - x0;
    const span = Math.abs(hi - lo);
    const tdec = span >= 50 ? 0 : span >= 5 ? 1 : 2;
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
      <text class="tt" x="${m.l}" y="15">${title}</text>
      <text class="cv" x="${w - m.r}" y="15" text-anchor="end">${this.fmt(cur, dec(cur))} ${unit}</text>
      ${niceTicks(lo, hi, 3).map((t) => `<line class="gr" x1="${m.l}" x2="${w - m.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}"/><text class="tk" x="${m.l - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end">${this.fmt(t, tdec)}</text>`).join("")}
      ${tx.map((t) => `<text class="tk" x="${X(t).toFixed(1)}" y="${h - 6}" text-anchor="middle">${this.fmt(t + c0, Math.abs(x1 - x0) < 8 ? (Math.abs(x1 - x0) < 0.8 ? 3 : 2) : 0)}</text>`).join("")}
      <path d="${area}" class="ar"/><path d="${line}" class="ln"/>
      <line class="zr" x1="${m.l}" x2="${w - m.r}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}"/>
      ${xs.length && dmax > 0 ? ext(imax, !P.flip) : ""}${xs.length && dmin < 0 ? ext(imin, !!P.flip) : ""}
      <line class="cu" x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${m.t - 4}" y2="${h - m.b}"/>
      <circle cx="${X(v).toFixed(1)}" cy="${Y(cur).toFixed(1)}" r="4" class="pt" stroke="var(--panel)" stroke-width="1.5"/></svg>`;
  }

  private renderDock() {
    const sel = this.sel;
    if (!sel) return;
    const f = this.forcesAt(sel.v), st = this.inp.results[this.step];
    const chip = (k: string, val: string) => `<span class="vis-chip"><span class="q">${k}</span> ${val}</span>`;
    const coord = this.shownCoord(sel.v);
    const lu = this.inp.units.model("length");
    let head: string, note: string;
    if (sel.kind === "member") {
      const path = this.paths[sel.index], part = this.inp.vm.parts[path.m.part];
      head = `<b>${part.name}</b> <span class="vis-pr">${this.partDesc(part)} · member ${sel.index + 1} · ${st?.name ?? ""} · cut at s = ${this.fmt(sel.v, 2)}${lu ? " " + lu : ""} of ${this.fmt(path.m.length, 2)}</span>`;
      note = `Drag the cut in the model or in a diagram to move it. You see the positive section: <i>x</i> along the member (towards you), <i>y</i> to the left, <i>z</i> downwards. The moment diagram is drawn on the tension side.${this.F?.members[sel.index]?.derivedShear ? " The shear forces are the slope of the moments (the elements give none)." : ""}`;
      this.ui.axisRow.innerHTML = "";
    } else {
      const body = this.inp.vm.bodies[sel.index];
      const parts = [...new Set(sel.recs.map((r) => this.inp.vm.parts[r.group.part].name))].join(", ");
      head = `<b>${parts}</b> <span class="vis-pr">${body.dim === 3 ? "solid" : "planar"} body · ${body.elements} elements · ${st?.name ?? ""} · cut at ${coord.name} = ${this.fmt(coord.v, 3)}${lu ? " " + lu : ""}${sel.cut ? ` · area ${this.fmt(sel.cut.area, 4)}` : ""}</span>`;
      const method = sel.cut?.method === "stresses" ? "the stresses integrated over the cut (nonlinear model: approximate, recovered from the element means)" : "the element nodal forces of a free-body cut (exact for the FE solution)";
      note = `Section forces from ${method}, about the centroid of the cut. Drag the cut or a diagram to move it. <i>x</i> along the axis, <i>y</i> to the left, <i>z</i> downwards.`;
      const ax = sel.axisMode ?? "auto";
      this.ui.axisRow.innerHTML = `Axis ${(["auto", "x", "y", "z"] as const).map((k) => `<button type="button" class="${k === ax ? "on" : ""}" data-axis="${k}">${k === "auto" ? "main" : k}</button>`).join("")}${sel.diag ? "" : ' <span class="vis-pr">integrating …</span>'}`;
      this.ui.axisRow.querySelectorAll<HTMLButtonElement>("[data-axis]").forEach(
        (b) =>
          (b.onclick = () => {
            const P = this.cutPoint();
            this.selectBody(sel.rec, P, false, b.dataset.axis as Sel["axisMode"]);
          }),
      );
    }
    this.ui.dockHead.innerHTML = head;
    this.ui.dockNote.innerHTML = note;
    let defl = "";
    if (this.U) {
      let d: THREE.Vector3 | null = null;
      if (sel.kind === "member") {
        const fa = this.frameAt(this.paths[sel.index], sel.v);
        d = this.pathDisp(this.paths[sel.index], fa.i, fa.w, V3(0, 0, 0));
      } else if (sel.cut) {
        const n = this.nearestNode(V3(...sel.cut.centroid), sel.index);
        if (n >= 0) d = vec(this.U, 3 * n);
      }
      if (d) defl = chip("deflection", this.q(-d.y, "length"));
    }
    this.ui.dockVals.innerHTML =
      chip(sym("N"), this.q(f[0], "force")) +
      chip(sym("V", "y"), this.q(f[1], "force")) +
      chip(sym("V", "z"), this.q(f[2], "force")) +
      chip(sym("M", "y"), this.q(f[4], "moment")) +
      chip(sym("M", "z"), this.q(f[5], "moment")) +
      chip(sym("T"), this.q(f[3], "moment")) +
      defl;
    const box = this.ui.plots;
    if (!this.selForces()) {
      box.innerHTML = "";
      return;
    }
    const cols = box.clientWidth < 620 ? 1 : 3;
    this.plotW = Math.floor((box.clientWidth - (cols - 1) * 14) / cols);
    this.plotH = cols === 1 ? 108 : 150;
    const mx = this.maxForces();
    const pv = mx[1] > 2.5 * mx[2] ? { c: 1, s: sym("V", "y") } : { c: 2, s: sym("V", "z") };
    const pm = mx[5] > 2.5 * mx[4] ? { c: 5, s: sym("M", "z") } : { c: 4, s: sym("M", "y") };
    box.innerHTML = [
      { c: 0, name: "Normal", s: sym("N"), q: "force" as ConvQ },
      { name: "Shear", q: "force" as ConvQ, ...pv },
      { name: "Moment", q: "moment" as ConvQ, flip: pm.c === 4, ...pm },
    ]
      .map((P) => this.plotSVG(P, sel.v))
      .join("");
  }

  // ------------------------------------------------------------ supports: click = replace by the reactions
  private buildReactions() {
    this.reactGroup.clear();
    this.root.querySelectorAll(".vis-lblR").forEach((l) => l.remove());
    const F = this.F;
    if (!F) return;
    const fmax = Math.max(1e-30, ...F.supports.flatMap((s) => s.f.map(Math.abs)));
    const mmax = Math.max(1e-30, ...F.supports.flatMap((s) => s.m.map(Math.abs)));
    const L = this.R * 0.25, r = this.R * 0.005;
    for (const S of this.supports) {
      S.recs.forEach((rc) => (rc.mesh.visible = !S.open));
      S.g = null;
      S.tips = [];
      if (!S.open) continue;
      const sp = F.supports[S.index];
      if (!sp) continue;
      const g = new THREE.Group();
      g.userData.support = S.index;
      this.reactGroup.add(g);
      S.g = g;
      ["Rx", "Ry", "Rz"].forEach((nm, c) => {
        const f = sp.f[c];
        if (Math.abs(f) < 0.005 * fmax) return;
        // the force pushes on the support point: the head is at the point
        const dir = V3(0, 0, 0).setComponent(c, Math.sign(f)), len = (0.45 + (1.1 * Math.abs(f)) / fmax) * L * 0.5, tail = dir.clone().multiplyScalar(-len - 0.08 * L);
        g.add(arrow(tail, dir.clone().multiplyScalar(-0.08 * L), FORCE, r * 1.3));
        const l = el("div", "vis-lbl vis-lblR", `${sym("R", nm[1])} = ${this.q(f, "force")}`);
        this.root.appendChild(l);
        S.tips.push([tail, l]);
      });
      ["Mx", "My", "Mz"].forEach((nm, c) => {
        const m = sp.m[c];
        if (Math.abs(m) < 0.005 * mmax || Math.abs(m) < 1e-9 * fmax * this.R) return;
        const dir = V3(0, 0, 0).setComponent(c, Math.sign(m)), len = (0.35 + (0.6 * Math.abs(m)) / mmax) * L * 0.5;
        const from = dir.clone().multiplyScalar(0.1 * L), to = dir.clone().multiplyScalar(0.1 * L + len);
        g.add(arrow(from, to, AXIS, r, 2));
        const l = el("div", "vis-lbl vis-lblR", `${sym("M", nm[1])} = ${this.q(m, "moment")}`);
        this.root.appendChild(l);
        S.tips.push([to, l]);
      });
      g.add(new THREE.Mesh(new THREE.SphereGeometry(r * 2.4, 16, 12), new THREE.MeshBasicMaterial({ color: FORCE })));
    }
  }

  private updateReactions() {
    if (!this.F) return;
    const w = this.root.clientWidth, h = this.root.clientHeight;
    for (const S of this.supports) {
      if (!S.open || !S.g) continue;
      const sp = this.F.supports[S.index];
      const P = V3(...sp.center);
      if (this.U && sp.nodes.length) {
        let d = V3(0, 0, 0);
        for (const n of sp.nodes) d.add(vec(this.U, 3 * n));
        P.addScaledVector(d.multiplyScalar(1 / sp.nodes.length), this.curK());
      }
      S.g.position.copy(P);
      for (const [tail, l] of S.tips) {
        const q = tail.clone().add(P).project(this.camera);
        l.style.display = q.z < 1 ? "block" : "none";
        l.style.left = ((q.x + 1) / 2) * w + 8 + "px";
        l.style.top = ((1 - q.y) / 2) * h - 10 + "px";
      }
    }
  }

  private toggleSupport(i: number) {
    const S = this.supports[i];
    S.open = !S.open;
    this.buildReactions();
    if (!S.open) {
      if (this.pick?.kind === "support" && this.pick.index === i) this.pick = null;
    } else this.pick = { kind: "support", index: i };
    this.showInfo();
  }

  private supportInfo(i: number) {
    const sp = this.F!.supports[i], st = this.inp.results[this.step];
    const sum = [0, 1, 2].map((c) => this.F!.supports.reduce((a, s) => a + s.f[c], 0));
    const c = sp.center;
    const rows = ["x", "y", "z"].map((k, j) => `<tr><td>R${k}</td><td>${this.q(sp.f[j], "force")}</td><td>${this.q(sum[j], "force")}</td></tr>`).join("");
    const mrows = sp.m.some((m) => Math.abs(m) > 1e-9 * Math.max(1e-30, ...sp.f.map(Math.abs)) * this.R)
      ? ["x", "y", "z"].map((k, j) => `<tr><td>M${k}</td><td>${this.q(sp.m[j], "moment")}</td><td></td></tr>`).join("")
      : "";
    return `<span class="vis-x" data-close>✕</span><b>Support ${i + 1} – ${this.supportKind(sp.mask, sp.nodes.length)}</b><br>
      <span class="vis-pr">${sp.nodes.length} node${sp.nodes.length === 1 ? "" : "s"} · centre (${this.fmt(c[0], 2)}, ${this.fmt(c[1], 2)}, ${this.fmt(c[2], 2)}) · click the arrows to show the support again</span>
      <table><tr class="hd"><td>${st?.name ?? ""}</td><td>This support</td><td>All supports</td></tr>${rows}${mrows}</table>
      <div class="vis-pr">The reactions as they act on the structure${sp.nodes.length > 1 ? " (sum over the nodes, moments about the centre)" : ""}. Positive Ry is upwards.</div>`;
  }

  private showInfo() {
    const info = this.ui.info;
    const p = this.pick;
    if (!p) {
      info.style.display = "none";
      return;
    }
    info.innerHTML = p.kind === "support" ? this.supportInfo(p.index) : p.html;
    info.style.display = "block";
    const x = info.querySelector("[data-close]") as HTMLElement | null;
    if (x)
      x.onclick = () => {
        this.pick = null;
        this.showInfo();
      };
  }

  private async partInfo(rec: Rec, p: THREE.Vector3, face: number) {
    const part = this.inp.vm.parts[rec.group.part];
    const st = this.inp.results[this.step];
    let h = `<span class="vis-x" data-close>✕</span><b>${part.name}</b><br>${this.partDesc(part)}${part.material ? ` · ${part.material}` : ""}<br>
      <span class="vis-pr">(${this.fmt(p.x, 3)}, ${this.fmt(p.y, 3)}, ${this.fmt(p.z, 3)})</span>`;
    const mp = rec.map[0];
    if (this.U && "node" in mp) {
      // the nearest node of the hit triangle
      let best = mp.node[3 * face], bd = Infinity;
      for (let j = 0; j < 3; j++) {
        const n = mp.node[3 * face + j], q = vec(this.inp.mesh.positions, 3 * n), d = q.distanceToSquared(p);
        if (d < bd) {
          bd = d;
          best = n;
        }
      }
      const U = this.U;
      h += `<table><tr class="hd"><td>${st?.name ?? ""}</td><td></td></tr>
        <tr><td>U1 / U2 / U3</td><td>${this.q(U[3 * best], "length")} / ${this.q(U[3 * best + 1], "length")} / ${this.q(U[3 * best + 2], "length")}</td></tr>`;
      if (rec.kind === "shell") {
        const e = this.elementAt(rec, face);
        if (e >= 0) {
          try {
            const sh = await this.inp.engine.visualShell(this.inp.id, this.step, this.frame, e);
            const fu = this.unit("force"), mu = this.unit("moment"), lu = this.unit("length");
            const pf = (v: number) => this.fmt(this.qv(v, "force") / convertValue(1, "length", this.inp.units.model("length"), this.inp.units.model("length") === "mm" ? "mm" : "m"), 2);
            const pm = (v: number) => this.fmt(this.qv(v, "moment") / convertValue(1, "length", this.inp.units.model("length"), this.inp.units.model("length") === "mm" ? "mm" : "m"), 2);
            const per = this.inp.units.model("length") ? `/${this.inp.units.model("length") === "mm" ? "mm" : "m"}` : "";
            void lu;
            if (sh.sf)
              h += `<tr><td>Membrane N11 / N22</td><td>${pf(sh.sf[0])} / ${pf(sh.sf[1])} ${fu}${per}</td></tr>
                <tr><td>Shear N12</td><td>${pf(sh.sf[2])} ${fu}${per}</td></tr>
                <tr><td>Transverse V13 / V23</td><td>${pf(sh.sf[3])} / ${pf(sh.sf[4])} ${fu}${per}</td></tr>`;
            if (sh.sm)
              h += `<tr><td>Bending M11 / M22</td><td>${pm(sh.sm[0])} / ${pm(sh.sm[1])} ${mu}${per}</td></tr>
                <tr><td>Twisting M12</td><td>${pm(sh.sm[2])} ${mu}${per}</td></tr>`;
            const d1 = sh.dir1.map((x) => this.fmt(x, 2)).join(", ");
            h += `</table><div class="vis-pr">Element ${e + 1}: section forces per unit width (element means) in the shell's local directions, 1 = (${d1}).</div>`;
          } catch {
            h += "</table>";
          }
        } else h += "</table>";
      } else h += "</table>";
    }
    this.pick = { kind: "info", html: h };
    this.showInfo();
  }

  /** The element of a hit triangle of a shell slab (top and bottom faces only). */
  private elementAt(rec: Rec, face: number): number {
    const { tris, triElem } = this.inp.mesh;
    const mp = rec.map[0] as { node: Int32Array };
    const a = mp.node[3 * face], b = mp.node[3 * face + 1], c = mp.node[3 * face + 2];
    const set = new Set([a, b, c]);
    for (let k = 0; k < triElem.length; k++) {
      if (this.inp.vm.elemPart[triElem[k]] !== rec.group.part) continue;
      if (set.has(tris[3 * k]) && set.has(tris[3 * k + 1]) && set.has(tris[3 * k + 2])) return triElem[k];
    }
    // a side wall: any element of the part using two of the nodes
    for (let k = 0; k < triElem.length; k++) {
      if (this.inp.vm.elemPart[triElem[k]] !== rec.group.part) continue;
      let n = 0;
      for (let j = 0; j < 3; j++) if (set.has(tris[3 * k + j])) n++;
      if (n >= 2) return triElem[k];
    }
    return -1;
  }

  private refreshSelection() {
    this.buildReactions();
    if (this.pick?.kind === "support") this.showInfo();
    else if (this.pick) {
      this.pick = null;
      this.showInfo();
    }
    const sel = this.sel;
    if (sel) {
      if (sel.kind === "body") {
        const P = V3(...this.inp.vm.bodies[sel.index].center).addScaledVector(sel.axis!, sel.v);
        const mode = sel.axisMode;
        this.clearSel();
        const rec = this.groups.flatMap((o) => o.recs).find((r) => r.kind === "body" && r.body === sel.index);
        if (rec) this.selectBody(rec, P, false, mode);
      } else {
        sel.dirty = true;
        this.renderDock();
      }
    }
  }

  // ------------------------------------------------------------ pointer
  private setRay(e: PointerEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, (-(e.clientY - r.top) / r.height) * 2 + 1), this.camera);
  }

  private grabTargets(): THREE.Object3D[] {
    const sel = this.sel!;
    return [sel.cap, sel.disc];
  }

  private cutFromRay(): number {
    const sel = this.sel!;
    const O = this.ray.ray.origin, D = this.ray.ray.direction;
    if (sel.kind === "body") {
      // the point on the axis line closest to the ray
      const ax = sel.axis!, A = this.cutPoint().addScaledVector(ax, -sel.v);
      const w = A.clone().sub(O), b = ax.dot(D), d = ax.dot(w), e = D.dot(w), den = 1 - b * b;
      return den < 1e-9 ? sel.v : (b * e - d) / den;
    }
    const path = this.paths[sel.index], s = path.m.s;
    let best = sel.v, bd = Infinity;
    const Q = s.map((v) => this.deformedOnPath(path, v));
    for (let i = 0; i < Q.length - 1; i++) {
      const A = Q[i], u = Q[i + 1].clone().sub(A), w = A.clone().sub(O), a = u.dot(u), b = u.dot(D), d = u.dot(w), e = D.dot(w), den = a - b * b;
      const t = den < 1e-20 ? 0 : Math.min(1, Math.max(0, (b * e - d) / den)), X = A.clone().addScaledVector(u, t).sub(O);
      const dist = X.addScaledVector(D, -X.dot(D)).length();
      if (dist < bd) {
        bd = dist;
        best = s[i] + t * (s[i + 1] - s[i]);
      }
    }
    return best;
  }

  /** The member whose (deformed) axis passes closest to the ray, within ~6 px or its profile. */
  private pickMember(e: PointerEvent): { member: number; s: number; depth: number; tol: number } | null {
    const O = this.ray.ray.origin, D = this.ray.ray.direction;
    const h = this.renderer.domElement.getBoundingClientRect().height || 1;
    const pxAt = (depth: number) => (2 * depth * Math.tan((this.camera.fov * Math.PI) / 360)) / h;
    void e;
    let best: { member: number; s: number; depth: number; tol: number } | null = null, bd = Infinity;
    for (const o of this.groups) {
      if (!o.g.visible) continue;
      for (const r of o.recs) {
        if (r.kind !== "member" || !r.mesh.visible) continue;
        const path = this.paths[r.member!], s = path.m.s;
        const prof = this.inp.vm.profiles[this.inp.vm.parts[path.m.part].profile];
        // the profile's centroid (sections may be offset from the node line) and its radius about it
        const pp = prof ? (prof.outer.length ? prof.outer : prof.pieces.flat()) : [];
        const cx = pp.reduce((a, q) => a + q[0], 0) / (pp.length || 1), cy = pp.reduce((a, q) => a + q[1], 0) / (pp.length || 1);
        const rad = pp.reduce((a, q) => Math.max(a, Math.hypot(q[0] - cx, q[1] - cy)), 0);
        // sample the path finely (it bends with the deformation)
        const pts: THREE.Vector3[] = [], ss: number[] = [];
        for (let i = 0; i < s.length - 1; i++) {
          const k = path.beam ? 6 : 1;
          for (let j = i === 0 ? 0 : 1; j <= k; j++) {
            const v = s[i] + ((s[i + 1] - s[i]) * j) / k;
            ss.push(v);
            const fr = this.frameAt(path, v);
            pts.push(this.deformedOnPath(path, v).addScaledVector(fr.n1, cx).addScaledVector(fr.n2, cy));
          }
        }
        for (let i = 0; i < pts.length - 1; i++) {
          const A = pts[i], u = pts[i + 1].clone().sub(A), w = A.clone().sub(O), a = u.dot(u), b = u.dot(D), d = u.dot(w), ee = D.dot(w), den = a - b * b;
          const t = den < 1e-20 ? 0 : Math.min(1, Math.max(0, (b * ee - d) / den));
          const X = A.clone().addScaledVector(u, t);
          const depth = X.clone().sub(O).dot(D);
          if (depth <= 0) continue;
          const dist = X.clone().sub(O).addScaledVector(D, -depth).length();
          const tol = Math.max(rad * 1.05, 6 * pxAt(depth));
          if (dist > tol) continue;
          const score = dist / tol + depth * 1e-6;
          if (score < bd) {
            bd = score;
            best = { member: r.member!, s: ss[i] + t * (ss[i + 1] - ss[i]), depth: depth - rad, tol: rad + 6 * pxAt(depth) };
          }
        }
      }
    }
    return best;
  }

  private onDownCapture = (e: PointerEvent) => {
    if (!this.sel) return;
    this.setRay(e);
    if (!this.ray.intersectObjects(this.grabTargets(), true).length) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    this.dragging = true;
    this.down = null;
    this.renderer.domElement.setPointerCapture(e.pointerId);
    this.renderer.domElement.style.cursor = "grabbing";
  };
  private onMove = (e: PointerEvent) => {
    if (this.dragging) {
      this.setRay(e);
      this.setCut(this.cutFromRay());
      return;
    }
    if (this.sel && !e.buttons) {
      this.setRay(e);
      this.renderer.domElement.style.cursor = this.ray.intersectObjects(this.grabTargets(), true).length ? "grab" : "";
    }
  };
  private onUpCapture = (e: PointerEvent) => {
    if (!this.dragging) return;
    this.dragging = false;
    this.renderer.domElement.releasePointerCapture(e.pointerId);
    this.renderer.domElement.style.cursor = "";
    e.stopImmediatePropagation();
  };
  private onDown = (e: PointerEvent) => {
    this.down = [e.clientX, e.clientY];
  };
  private onUp = (e: PointerEvent) => {
    if (!this.down || Math.hypot(e.clientX - this.down[0], e.clientY - this.down[1]) > 4) return;
    this.setRay(e);
    const ra = this.ray.intersectObjects(this.reactGroup.children, true)[0];
    if (ra) {
      let q: THREE.Object3D | null = ra.object;
      while (q && q.userData.support === undefined) q = q.parent;
      if (q) {
        this.toggleSupport(q.userData.support);
        return;
      }
    }
    const meshes = this.groups.filter((o) => o.g.visible).flatMap((o) => o.recs.filter((r) => r.mesh.visible).map((r) => r.mesh));
    for (const m of meshes) {
      m.geometry.computeBoundingSphere();
      m.geometry.boundingBox = null;
    }
    const own = this.sel ? this.ray.intersectObjects(this.sel.recs.map((r) => r.mesh))[0] : null;
    const hit = own || this.ray.intersectObjects(meshes)[0];
    // members: the closest member axis within a few pixels (or within its profile) of the ray
    const mp = this.pickMember(e);
    const hitKind = hit ? (hit.object.userData.rec as Rec).kind : null;
    if (this.F && mp && hitKind !== "member" && (!hit || mp.depth <= hit.distance) && !own) {
      const rec = this.groups.flatMap((o) => o.recs).find((r) => r.kind === "member" && r.member === mp.member)!;
      this.selectMember(rec, mp.s, !this.sel || this.sel.kind !== "member" || this.sel.index !== mp.member);
      return;
    }
    if (!hit) {
      this.pick = null;
      this.showInfo();
      this.clearSel();
      return;
    }
    const rec = hit.object.userData.rec as Rec;
    // the undeformed position of the hit point
    const fi = hit.faceIndex ?? 0, base = rec.base[0], i = 9 * fi;
    const bary = hit.barycoord ?? V3(1 / 3, 1 / 3, 1 / 3);
    const p = V3(0, 0, 0);
    for (let j = 0; j < 3; j++) p.addScaledVector(V3(base[i + 3 * j], base[i + 3 * j + 1], base[i + 3 * j + 2]), [bary.x, bary.y, bary.z][j]);
    if (rec.kind === "support" && this.F && rec.support !== undefined) {
      this.toggleSupport(rec.support);
      return;
    }
    if (this.F && rec.kind === "member") {
      const path = this.paths[rec.member!], mp = rec.map[0] as { seg: Int32Array; xi: Float32Array };
      const v0 = 3 * fi;
      let s = 0;
      for (let j = 0; j < 3; j++) {
        const sg = mp.seg[v0 + j], xi = mp.xi[v0 + j];
        s += (path.m.s[sg] + xi * (path.m.s[sg + 1] - path.m.s[sg])) / 3;
      }
      this.selectMember(rec, s, !this.sel || this.sel.kind !== "member" || this.sel.index !== rec.member);
      return;
    }
    if (this.F && rec.kind === "body") {
      this.selectBody(rec, p, !this.sel || this.sel.kind !== "body" || this.sel.index !== rec.body);
      return;
    }
    this.clearSel();
    this.partInfo(rec, p, fi);
  };
  private onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      this.clearSel();
      this.pick = null;
      this.showInfo();
    }
  };

  // ------------------------------------------------------------ styles
  private static injectCss() {
    if (document.getElementById(CSS_ID)) return;
    const s = document.createElement("style");
    s.id = CSS_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }
}

const CSS = `
.vis { position:absolute; inset:0; overflow:hidden; --plot:#2F5597; --plotfill:#DCE6F2; font-size:13px; line-height:1.45; color:var(--text); }
.vis.dark { --plot:#8fb0ea; --plotfill:rgba(143,176,234,.2); }
.vis canvas { display:block; }
.vis .vis-panel { position:absolute; top:12px; left:12px; width:300px; max-height:calc(100% - 24px); overflow:auto; background:var(--overlay);
  border:1px solid var(--line); border-radius:10px; padding:12px 14px; box-sizing:border-box; backdrop-filter:blur(6px); }
.vis h1 { font-size:15px; margin:0 0 2px; font-weight:650; word-break:break-word; }
.vis .vis-sub { color:var(--muted); margin:0 0 10px; }
.vis .vis-row { display:grid; grid-template-columns:16px 12px 1fr; gap:8px; align-items:start; padding:5px 0; border-top:1px solid var(--line); cursor:pointer; }
.vis .vis-row input { margin:2px 0 0; }
.vis .vis-sw { width:12px; height:12px; border-radius:3px; margin-top:3px; }
.vis .vis-nm { font-weight:600; }
.vis .vis-pr { color:var(--muted); font-variant-numeric:tabular-nums; }
.vis .vis-opts { border-top:1px solid var(--line); padding-top:8px; margin-top:2px; display:grid; gap:4px; }
.vis .vis-res { display:grid; gap:4px; }
.vis .vis-opts label { cursor:pointer; }
.vis .vis-opts input[type=range] { vertical-align:middle; width:130px; }
.vis .vis-views { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
.vis button { font:inherit; color:var(--text); background:transparent; border:1px solid var(--line); border-radius:6px; padding:3px 9px; cursor:pointer; }
.vis button:hover { border-color:var(--muted); }
.vis button.on { background:var(--accent-bg); border-color:var(--accent); }
.vis select { font:inherit; color:var(--text); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:2px 4px; max-width:190px; }
.vis .vis-info { position:absolute; right:12px; bottom:12px; background:var(--overlay); border:1px solid var(--line); border-radius:8px; padding:8px 12px; width:320px;
  max-height:calc(100% - 24px); overflow:auto; box-sizing:border-box; display:none; backdrop-filter:blur(6px); }
.vis .vis-info table { border-collapse:collapse; width:100%; margin-top:4px; font-variant-numeric:tabular-nums; }
.vis .vis-info td { padding:1px 0 1px 8px; } .vis .vis-info td:first-child { padding-left:0; } .vis .vis-info td:nth-child(n+2) { text-align:right; }
.vis .vis-info .hd td { color:var(--muted); border-bottom:1px solid var(--line); }
.vis .vis-x { float:right; cursor:pointer; color:var(--muted); }
.vis .vis-dock { position:absolute; left:324px; right:12px; bottom:12px; background:var(--overlay); border:1px solid var(--line); border-radius:10px; padding:10px 14px; display:none; backdrop-filter:blur(6px); }
.vis .vis-dh { display:flex; justify-content:space-between; gap:12px; }
.vis .vis-axisrow { margin-top:4px; display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
.vis .vis-axisrow:empty { display:none; }
.vis .vis-axisrow button { padding:1px 8px; }
.vis .vis-dockvals { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 4px; }
.vis .vis-chip { border:1px solid var(--line); border-radius:999px; padding:1px 9px; font-variant-numeric:tabular-nums; }
.vis .vis-plots { display:flex; flex-wrap:wrap; gap:4px 14px; cursor:ew-resize; touch-action:none; user-select:none; overflow:hidden; }
.vis .vis-plots svg { flex:none; display:block; }
.vis .vis-plots text { font:11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; fill:var(--muted); font-variant-numeric:tabular-nums; }
.vis .vis-plots .tt { fill:var(--text); font-weight:600; font-size:12px; } .vis .vis-plots .cv { font-weight:700; font-size:12px; fill:var(--plot); }
.vis .vis-plots .it { font:italic 15px "Times New Roman", Times, serif; } .vis .vis-plots .sb { font:10px "Times New Roman", Times, serif; }
.vis .vis-plots .ex { fill:var(--text); font:11.5px "Times New Roman", Times, serif; }
.vis .vis-plots .ar { fill:var(--plotfill); } .vis .vis-plots .ln { fill:none; stroke:var(--plot); stroke-width:1.6; stroke-linejoin:round; }
.vis .vis-plots .gr { stroke:var(--line); stroke-width:1; } .vis .vis-plots .zr { stroke:var(--muted); stroke-width:1; }
.vis .vis-plots .cu { stroke:var(--plot); stroke-width:1; stroke-dasharray:3 3; } .vis .vis-plots .pt { fill:var(--plot); }
.vis .vis-lbl { position:absolute; pointer-events:none; white-space:nowrap; font-variant-numeric:tabular-nums; font-size:12px; color:#2F5597;
  text-shadow:0 0 3px var(--bg), 0 0 3px var(--bg), 0 0 3px var(--bg); }
.vis .vis-lbl i { font-family:"Times New Roman", Times, serif; font-size:15px; }
.vis .vis-lbl-ax { color:#A3283F; }
.vis.dark .vis-lbl { color:#8fb0ea; } .vis.dark .vis-lbl-ax { color:#e07a8c; }
.vis .vis-chip .q { font-family:"Times New Roman", Times, serif; font-size:15px; color:#2F5597; margin-right:3px; }
.vis.dark .vis-chip .q { color:#8fb0ea; }
.vis .vis-chip sub, .vis .vis-lbl sub { font-size:10px; }
.vis .vis-hint { position:absolute; left:324px; bottom:12px; color:var(--muted); pointer-events:none; }
.vis .vis-tips { position:absolute; inset:0; pointer-events:none; }
@media (max-width:900px) { .vis .vis-dock { left:12px; max-height:58%; overflow:auto; } .vis .vis-hint { display:none; } }
@media (max-width:640px) { .vis .vis-panel { width:calc(100% - 24px); max-height:45%; } }
`;
