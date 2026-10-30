//! Kopplingselementens kinematik med stora förskjutningar och rotationer
//! (NLGEOM, inkluderas med `#[path]` från `connector.rs`).
//!
//! Med NLGEOM räknas varje komponent som en exakt (icke-linjäriserad)
//! funktion av nodernas lägen x_a, x_b och rotationer R_a, R_b. De lokala
//! axlarna roterar med noderna: a_i = R_a e_i^a, b_i = R_b e_i^b,
//! d = x_b − x_a och ψ = log(R_b R_aᵀ) (den relativa rotationsvektorn,
//! rumsliga komponenter). Komponentens värde är måttet minus måttet i
//! referensläget:
//!
//! - CARTESIAN, JOIN, SLOT, SLIDE-PLANE: a_i · d. PROJECTION CARTESIAN:
//!   a_i^m · d med mittsystemet a_i^m = exp(½ψ) a_i.
//! - AXIAL, LINK: |d|. RADIAL-THRUST: √((a_1·d)² + (a_2·d)²) och a_3 · d.
//! - ALIGN, ROTATION, PROJECTION FLEXION-TORSION: a_i · ψ (rotationsvektorns
//!   komponenter i a:s system); FLEXION-TORSION i ordningen (2, 3, 1).
//! - REVOLUTE och UNIVERSAL: den relativa rotationen Q = Aᵀ B (Q_ij = a_i·b_j)
//!   delas upp som Q = R_x(α) R_z(γ): komponent 4 = α = atan2(−Q₂₃, Q₃₃),
//!   UNIVERSAL komponent 6 = γ = atan2(−Q₁₂, Q₁₁); villkoren är
//!   Q₁₃ = a_1 · b_3 = 0 (båda) och −Q₁₂ = 0 (REVOLUTE).
//! - CONSTANT VELOCITY: −ψ · ŝ, ŝ = (a_3 + b_3)/|a_3 + b_3| (ingen vridning
//!   kring bisektrisen).
//! - CARDAN och EULER: vinklarna ur Q med samma uppdelning som i referensläget
//!   (`connector_kin.rs`), utan linjärisering.
//!
//! För små rörelser ger måtten exakt raderna i `connector_kin.rs`.
//! Gradienterna mot translationerna räknas analytiskt och mot rotationerna
//! (rumsliga spinn δw, R ← exp([δw]×) R som i `steps/nlgeom.rs`) med
//! fyrpunkts centraldifferenser; den geometriska styvheten Σ F ∂²g/∂q² (behövs för
//! kvadratisk konvergens vid stora rotationer) med differenser av
//! gradienten. Vinkelmått följs kontinuerligt förbi ±π (närmast värdet vid
//! föregående godkända inkrement).

use crate::fem::dof::{Dof, NodeVec, U1, UR1, UR3};
use crate::kinematics::tensor::{self, M3};

type V3 = [f64; 3];

/// Måttet för en komponent med stora rotationer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    None,
    /// a_i · d
    Trans(usize),
    /// a_i^m · d (mittsystemet)
    TransMid(usize),
    /// |d|
    Length,
    /// √((a_1·d)² + (a_2·d)²)
    Radial,
    /// a_i · ψ
    RotVec(usize),
    /// α = atan2(−a_2·b_3, a_3·b_3)
    Alpha,
    /// γ = atan2(−a_1·b_2, a_1·b_1)
    Gamma,
    /// s · (a_i · b_j)
    Dot(usize, usize, f64),
    /// −ψ · ŝ
    Cv,
    Cardan(usize),
    Euler(usize),
}

impl Measure {
    fn angular(&self) -> bool {
        matches!(self, Measure::Alpha | Measure::Gamma | Measure::Cardan(_) | Measure::Euler(_))
    }
}

/// Måtten för kopplingstyperna (samma uppdelning som `kin::components`).
pub fn measures(types: &[String]) -> [Measure; 6] {
    use Measure::*;
    let mut m = [None; 6];
    let set_trans = |m: &mut [Measure; 6], name: &str| match name {
        "CARTESIAN" | "JOIN" | "SLOT" | "SLIDE-PLANE" => {
            for i in 0..3 {
                m[i] = Trans(i);
            }
        }
        "PROJECTION CARTESIAN" => {
            for i in 0..3 {
                m[i] = TransMid(i);
            }
        }
        "LINK" | "AXIAL" => m[0] = Length,
        "RADIAL-THRUST" => {
            m[0] = Radial;
            m[2] = Trans(2);
        }
        _ => {}
    };
    let set_rot = |m: &mut [Measure; 6], name: &str| match name {
        "ALIGN" | "ROTATION" | "PROJECTION FLEXION-TORSION" => {
            for i in 0..3 {
                m[3 + i] = RotVec(i);
            }
        }
        "FLEXION-TORSION" => {
            for (k, i) in [1, 2, 0].into_iter().enumerate() {
                m[3 + k] = RotVec(i);
            }
        }
        "REVOLUTE" => {
            m[3] = Alpha;
            m[4] = Dot(0, 2, 1.0);
            m[5] = Dot(0, 1, -1.0);
        }
        "UNIVERSAL" => {
            m[3] = Alpha;
            m[4] = Dot(0, 2, 1.0);
            m[5] = Gamma;
        }
        "CONSTANT VELOCITY" => m[4] = Cv,
        "CARDAN" => {
            for k in 0..3 {
                m[3 + k] = Cardan(k);
            }
        }
        "EULER" => {
            for k in 0..3 {
                m[3 + k] = Euler(k);
            }
        }
        _ => {}
    };
    for t in types {
        let (tr, ro) = match t.as_str() {
            "BEAM" | "WELD" => ("JOIN", "ALIGN"),
            "HINGE" => ("JOIN", "REVOLUTE"),
            "UJOINT" => ("JOIN", "UNIVERSAL"),
            "CVJOINT" => ("JOIN", "CONSTANT VELOCITY"),
            "CYLINDRICAL" => ("SLOT", "REVOLUTE"),
            "PLANAR" => ("SLIDE-PLANE", "REVOLUTE"),
            "TRANSLATOR" => ("SLOT", "ALIGN"),
            "BUSHING" => ("PROJECTION CARTESIAN", "PROJECTION FLEXION-TORSION"),
            other => (other, other),
        };
        set_trans(&mut m, tr);
        set_rot(&mut m, ro);
    }
    m
}

/// Ett elements geometri för NLGEOM-kinematiken.
#[derive(Debug, Clone)]
pub struct Geo {
    pub a: Option<usize>,
    pub b: Option<usize>,
    /// Referenslägen (en jordad nod har den andra nodens läge).
    pub xa: V3,
    pub xb: V3,
    pub ea: [V3; 3],
    pub eb: [V3; 3],
    /// Noderna har rotationsfrihetsgrader som används.
    pub ra: bool,
    pub rb: bool,
    pub dim: usize,
    pub meas: [Measure; 6],
    /// Måttens värden i referensläget.
    pub refv: [f64; 6],
    /// EULER med singulär referens (samma axel e3): vinklarna skrivs som 0.
    pub euler_singular: bool,
}

/// Ett läge: nodernas positioner och rotationer.
#[derive(Clone, Copy)]
struct Config {
    xa: V3,
    xb: V3,
    ra: M3,
    rb: M3,
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// En variabel: (nod, frihetsgrad, a- eller b-nod, translation, riktning).
#[derive(Clone, Copy)]
struct Var {
    node: usize,
    dof: Dof,
    at_b: bool,
    rot: bool,
    q: usize,
}

impl Geo {
    #[allow(clippy::too_many_arguments)]
    pub fn new(a: Option<usize>, b: Option<usize>, xa: V3, xb: V3, ea: [V3; 3], eb: [V3; 3], ra: bool, rb: bool, dim: usize, types: &[String]) -> Geo {
        let meas = measures(types);
        let r0 = (0..3).map(|i| (0..3).map(|j| dot(ea[i], eb[j])).collect::<Vec<_>>()).collect::<Vec<_>>();
        let euler_singular = r0[2][2].clamp(-1.0, 1.0).acos().sin().abs() < 1e-9;
        let mut g = Geo { a, b, xa, xb, ea, eb, ra, rb, dim, meas, refv: [0.0; 6], euler_singular };
        let c = Config { xa, xb, ra: tensor::I3, rb: tensor::I3 };
        g.refv = g.raw(&c);
        g
    }

    fn vars(&self) -> Vec<Var> {
        let mut v = Vec::new();
        let tq: &[usize] = if self.dim == 2 { &[0, 1] } else { &[0, 1, 2] };
        let rq: &[usize] = if self.dim == 2 { &[2] } else { &[0, 1, 2] };
        for (node, at_b, rok) in [(self.a, false, self.ra), (self.b, true, self.rb)] {
            let Some(n) = node else { continue };
            for &q in tq {
                v.push(Var { node: n, dof: U1 + q as Dof, at_b, rot: false, q });
            }
            if rok {
                for &q in rq {
                    v.push(Var { node: n, dof: UR1 + q as Dof, at_b, rot: true, q });
                }
            }
        }
        v
    }

    fn rotation(&self, u: &NodeVec) -> M3 {
        if self.dim == 2 {
            tensor::exp_so3([0.0, 0.0, u[(UR3 - 1) as usize]])
        } else {
            tensor::exp_so3([u[3], u[4], u[5]])
        }
    }

    fn config(&self, u: &[NodeVec]) -> Config {
        let ua = self.a.map(|n| u[n]).unwrap_or([0.0; 8]);
        let ub = self.b.map(|n| u[n]).unwrap_or([0.0; 8]);
        let xa = [self.xa[0] + ua[0], self.xa[1] + ua[1], self.xa[2] + ua[2]];
        let xb = [self.xb[0] + ub[0], self.xb[1] + ub[1], self.xb[2] + ub[2]];
        let ra = if self.ra && self.a.is_some() { self.rotation(&ua) } else { tensor::I3 };
        let rb = if self.rb && self.b.is_some() { self.rotation(&ub) } else { tensor::I3 };
        Config { xa, xb, ra, rb }
    }

    /// Måtten (utan referensvärdet) i läget `c`.
    fn raw(&self, c: &Config) -> [f64; 6] {
        let ax: [V3; 3] = std::array::from_fn(|i| tensor::mv(&c.ra, self.ea[i]));
        let bx: [V3; 3] = std::array::from_fn(|i| tensor::mv(&c.rb, self.eb[i]));
        let d = [c.xb[0] - c.xa[0], c.xb[1] - c.xa[1], c.xb[2] - c.xa[2]];
        let rel = tensor::mul_bt(&c.rb, &c.ra);
        let psi = tensor::log_so3(&rel);
        let q = |i: usize, j: usize| dot(ax[i], bx[j]);
        let mut out = [0.0; 6];
        for k in 0..6 {
            out[k] = match self.meas[k] {
                Measure::None => 0.0,
                Measure::Trans(i) => dot(ax[i], d),
                Measure::TransMid(i) => {
                    let h = tensor::exp_so3([0.5 * psi[0], 0.5 * psi[1], 0.5 * psi[2]]);
                    dot(tensor::mv(&h, ax[i]), d)
                }
                Measure::Length => norm(d),
                Measure::Radial => {
                    let (p1, p2) = (dot(ax[0], d), dot(ax[1], d));
                    (p1 * p1 + p2 * p2).sqrt()
                }
                Measure::RotVec(i) => dot(ax[i], psi),
                Measure::Alpha => (-q(1, 2)).atan2(q(2, 2)),
                Measure::Gamma => (-q(0, 1)).atan2(q(0, 0)),
                Measure::Dot(i, j, s) => s * q(i, j),
                Measure::Cv => {
                    let sv = [ax[2][0] + bx[2][0], ax[2][1] + bx[2][1], ax[2][2] + bx[2][2]];
                    let n = norm(sv).max(1e-300);
                    -dot(psi, sv) / n
                }
                Measure::Cardan(kk) => match kk {
                    0 => (-q(1, 2)).atan2(q(2, 2)),
                    1 => q(0, 2).clamp(-1.0, 1.0).asin(),
                    _ => (-q(0, 1)).atan2(q(0, 0)),
                },
                Measure::Euler(kk) => {
                    if self.euler_singular {
                        0.0
                    } else {
                        match kk {
                            0 => q(0, 2).atan2(-q(1, 2)),
                            1 => q(2, 2).clamp(-1.0, 1.0).acos(),
                            _ => q(2, 0).atan2(q(2, 1)),
                        }
                    }
                }
            };
        }
        out
    }

    /// Komponenternas värden (måttet minus referensvärdet; vinklar närmast `near`).
    fn values(&self, c: &Config, near: &[f64; 6]) -> [f64; 6] {
        let r = self.raw(c);
        let two_pi = 2.0 * std::f64::consts::PI;
        std::array::from_fn(|k| {
            let mut g = r[k] - self.refv[k];
            if self.meas[k].angular() {
                g += two_pi * ((near[k] - g) / two_pi).round();
            }
            g
        })
    }

    /// Läget med variabeln `v` störd med `e` (rotationer som rumsligt spinn).
    fn perturb(c: &Config, v: &Var, e: f64) -> Config {
        let mut c = *c;
        if v.rot {
            let mut w = [0.0; 3];
            w[v.q] = e;
            let r = tensor::exp_so3(w);
            if v.at_b {
                c.rb = tensor::mul(&r, &c.rb);
            } else {
                c.ra = tensor::mul(&r, &c.ra);
            }
        } else if v.at_b {
            c.xb[v.q] += e;
        } else {
            c.xa[v.q] += e;
        }
        c
    }

    /// Gradienten för varje komponent mot variablerna (translationer analytiskt,
    /// rotationer med centraldifferens).
    fn grad(&self, c: &Config, vars: &[Var], near: &[f64; 6]) -> [Vec<f64>; 6] {
        let mut gr: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; vars.len()]);
        let ax: [V3; 3] = std::array::from_fn(|i| tensor::mv(&c.ra, self.ea[i]));
        let d = [c.xb[0] - c.xa[0], c.xb[1] - c.xa[1], c.xb[2] - c.xa[2]];
        let rel = tensor::mul_bt(&c.rb, &c.ra);
        let psi = tensor::log_so3(&rel);
        // ∂g/∂x_b för translationsmåtten (∂g/∂x_a = −∂g/∂x_b)
        let tdir: [V3; 6] = std::array::from_fn(|k| match self.meas[k] {
            Measure::Trans(i) => ax[i],
            Measure::TransMid(i) => tensor::mv(&tensor::exp_so3([0.5 * psi[0], 0.5 * psi[1], 0.5 * psi[2]]), ax[i]),
            Measure::Length => {
                let l = norm(d).max(1e-300);
                [d[0] / l, d[1] / l, d[2] / l]
            }
            Measure::Radial => {
                let (p1, p2) = (dot(ax[0], d), dot(ax[1], d));
                let r = (p1 * p1 + p2 * p2).sqrt().max(1e-300);
                std::array::from_fn(|q| (p1 * ax[0][q] + p2 * ax[1][q]) / r)
            }
            _ => [0.0; 3],
        });
        // fyrpunkts centraldifferens (fel ~h⁴, avrundning ~10⁻¹³)
        const EPS: f64 = 1e-3;
        for (j, v) in vars.iter().enumerate() {
            if v.rot {
                let g1 = self.values(&Self::perturb(c, v, EPS), near);
                let g2 = self.values(&Self::perturb(c, v, 2.0 * EPS), near);
                let m1 = self.values(&Self::perturb(c, v, -EPS), near);
                let m2 = self.values(&Self::perturb(c, v, -2.0 * EPS), near);
                for k in 0..6 {
                    gr[k][j] = (8.0 * (g1[k] - m1[k]) - (g2[k] - m2[k])) / (12.0 * EPS);
                }
            } else {
                let s = if v.at_b { 1.0 } else { -1.0 };
                for k in 0..6 {
                    gr[k][j] = s * tdir[k][v.q];
                }
            }
        }
        gr
    }
}

/// Komponenter, rader och (vid behov) Hessianer i läget `u`.
pub struct Eval {
    pub g: [f64; 6],
    pub rows: [Vec<(usize, Dof, f64)>; 6],
    vars: Vec<Var>,
    cfg: Config,
    near: [f64; 6],
}

impl Geo {
    /// Utvärderar komponenterna i läget `u`; `near` = värdena vid föregående
    /// godkända inkrement (för vinklarnas kontinuitet).
    pub fn eval(&self, u: &[NodeVec], near: &[f64; 6]) -> Eval {
        let cfg = self.config(u);
        let vars = self.vars();
        let g = self.values(&cfg, near);
        let gr = self.grad(&cfg, &vars, &g);
        let rows = std::array::from_fn(|k| vars.iter().zip(&gr[k]).filter(|(_, v)| v.abs() > 1e-14).map(|(x, &v)| (x.node, x.dof, v)).collect());
        Eval { g, rows, vars, cfg, near: g }
    }

    /// Den geometriska styvheten Σ_k F_k ∂²g_k/∂q² (symmetriserad) som
    /// (nod, frihetsgrad, nod, frihetsgrad, värde), med differenser av gradienten.
    pub fn geometric(&self, ev: &Eval, f: &[f64; 6]) -> Vec<(usize, Dof, usize, Dof, f64)> {
        if f.iter().all(|&x| x == 0.0) {
            return Vec::new();
        }
        let nv = ev.vars.len();
        let d = [ev.cfg.xb[0] - ev.cfg.xa[0], ev.cfg.xb[1] - ev.cfg.xa[1], ev.cfg.xb[2] - ev.cfg.xa[2]];
        let lt = 1e-5 * norm(d).max(norm([self.xb[0] - self.xa[0], self.xb[1] - self.xa[1], self.xb[2] - self.xa[2]])).max(1e-3);
        let mut h = vec![0.0; nv * nv];
        for j in 0..nv {
            let e = if ev.vars[j].rot { 1e-5 } else { lt };
            let gp = self.grad(&Self::perturb(&ev.cfg, &ev.vars[j], e), &ev.vars, &ev.near);
            let gm = self.grad(&Self::perturb(&ev.cfg, &ev.vars[j], -e), &ev.vars, &ev.near);
            for k in 0..6 {
                if f[k] == 0.0 {
                    continue;
                }
                for i in 0..nv {
                    h[i * nv + j] += f[k] * (gp[k][i] - gm[k][i]) / (2.0 * e);
                }
            }
        }
        let mut out = Vec::new();
        for i in 0..nv {
            for j in 0..nv {
                let v = 0.5 * (h[i * nv + j] + h[j * nv + i]);
                if v != 0.0 {
                    out.push((ev.vars[i].node, ev.vars[i].dof, ev.vars[j].node, ev.vars[j].dof, v));
                }
            }
        }
        out
    }
}
