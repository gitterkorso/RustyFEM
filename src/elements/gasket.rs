//! Packningselement: GK3D8, GK3D6, GKPE4, GKAX4 (med membran och
//! tvärskjuvning), länkarna GK2D2, GKAX2, GK3D2 och varianterna med bara
//! tjockleksbeteende (GK3D8N, GK3D6N, GKPE4N, GKAX4N, GK2D2N, GKAX2N, GK3D2N).
//!
//! Konventioner:
//! - Undersidans noder först, sedan översidans i samma ordning (GK3D8: 1–4
//!   och 5–8, GK3D6: 1–3 och 4–6, GKPE4/GKAX4: 1–2 och 3–4, länkar: 1 och 2).
//! - Integrationspunkterna ligger i mittytans noder (nodvis integrering med
//!   vikten ∫Nₐ dA; axisymmetri ∫Nₐ 2πr ds; länkar area 1).
//! - Tjockleksriktningen n är mittytans normal (3D), mittlinjens normal
//!   z × t (2D) eller nod 1 → nod 2 (länkar). Stängningen är
//!   c = −(u_över − u_under)·n och E11 = c (inte delad med tjockleken).
//! - Lokala riktningar: 3D e₂ = global x projicerad på mittytan, e₃ = n × e₂;
//!   2D e₂ = z × n. Tvärskjuvningen γ₁₂ = (u_över − u_under)·e₂ (E12),
//!   S12 = G·γ₁₂.
//! - Membrantöjningen (E22, E33, E23) räknas i elementets mitt ur mittytans
//!   förskjutning ½(u_under + u_över); membrantjockleken är den initiala
//!   tjockleken (första fältet på *GASKET SECTION, standard 1). GKAX4: E33 =
//!   ringtöjningen u_r/r i mitten.
//!
//! Utdata: S (S11 = tryck) och E (E11 = stängning) med sex komponenter i 3D
//! och fyra (11, 22, 33, 12) i 2D.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::materials::{MatInput, StressMode};
use crate::model::{fail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Hex8,
    Wedge6,
    Plane4,
    Axi4,
    Link2D,
    LinkAxi,
    Link3D,
}

pub struct Gasket {
    name: &'static str,
    kind: Kind,
    /// Bara tjockleksbeteende (N-varianterna).
    thickness_only: bool,
}

const fn g(name: &'static str, kind: Kind, thickness_only: bool) -> Gasket {
    Gasket { name, kind, thickness_only }
}

pub static GK3D8: Gasket = g("GK3D8", Kind::Hex8, false);
pub static GK3D8N: Gasket = g("GK3D8N", Kind::Hex8, true);
pub static GK3D6: Gasket = g("GK3D6", Kind::Wedge6, false);
pub static GK3D6N: Gasket = g("GK3D6N", Kind::Wedge6, true);
pub static GKPE4: Gasket = g("GKPE4", Kind::Plane4, false);
pub static GKPE4N: Gasket = g("GKPE4N", Kind::Plane4, true);
pub static GKAX4: Gasket = g("GKAX4", Kind::Axi4, false);
pub static GKAX4N: Gasket = g("GKAX4N", Kind::Axi4, true);
pub static GK2D2: Gasket = g("GK2D2", Kind::Link2D, false);
pub static GK2D2N: Gasket = g("GK2D2N", Kind::Link2D, true);
pub static GKAX2: Gasket = g("GKAX2", Kind::LinkAxi, false);
pub static GKAX2N: Gasket = g("GKAX2N", Kind::LinkAxi, true);
pub static GK3D2: Gasket = g("GK3D2", Kind::Link3D, false);
pub static GK3D2N: Gasket = g("GK3D2N", Kind::Link3D, true);

pub static TYPES: &[&dyn ElementType] =
    &[&GK3D8, &GK3D8N, &GK3D6, &GK3D6N, &GKPE4, &GKPE4N, &GKAX4, &GKAX4N, &GK2D2, &GK2D2N, &GKAX2, &GKAX2N, &GK3D2, &GK3D2N];

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f64; 3]) -> Result<[f64; 3]> {
    let n = dot(a, a).sqrt();
    if n <= 0.0 {
        return fail("packningens tjockleksriktning kan inte beräknas (noderna sammanfaller)");
    }
    Ok([a[0] / n, a[1] / n, a[2] / n])
}

/// Mittytans formfunktioner och derivator i den naturliga punkten p.
fn surf_shape(kind: Kind, p: [f64; 2]) -> (Vec<f64>, Vec<[f64; 2]>) {
    let (x, y) = (p[0], p[1]);
    match kind {
        Kind::Hex8 => {
            let c = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let n = c.iter().map(|c| 0.25 * (1.0 + c[0] * x) * (1.0 + c[1] * y)).collect();
            let d = c.iter().map(|c| [0.25 * c[0] * (1.0 + c[1] * y), 0.25 * c[1] * (1.0 + c[0] * x)]).collect();
            (n, d)
        }
        Kind::Wedge6 => (vec![1.0 - x - y, x, y], vec![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
        Kind::Plane4 | Kind::Axi4 => (vec![0.5 * (1.0 - x), 0.5 * (1.0 + x)], vec![[-0.5, 0.0], [0.5, 0.0]]),
        _ => (vec![1.0], vec![[0.0, 0.0]]),
    }
}

/// Mittytans noder i naturliga koordinater och elementets mittpunkt.
fn surf_nodes(kind: Kind) -> (Vec<[f64; 2]>, [f64; 2]) {
    match kind {
        Kind::Hex8 => (vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], [0.0, 0.0]),
        Kind::Wedge6 => (vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [1.0 / 3.0, 1.0 / 3.0]),
        Kind::Plane4 | Kind::Axi4 => (vec![[-1.0, 0.0], [1.0, 0.0]], [0.0, 0.0]),
        _ => (vec![[0.0, 0.0]], [0.0, 0.0]),
    }
}

/// Gaussregel på mittytan (för vikterna ∫Nₐ dA).
fn surf_rule(kind: Kind) -> Vec<([f64; 2], f64)> {
    let g = 1.0 / 3f64.sqrt();
    match kind {
        Kind::Hex8 => [[-g, -g], [g, -g], [-g, g], [g, g]].iter().map(|p| (*p, 1.0)).collect(),
        Kind::Wedge6 => vec![([1.0 / 6.0, 1.0 / 6.0], 1.0 / 6.0), ([2.0 / 3.0, 1.0 / 6.0], 1.0 / 6.0), ([1.0 / 6.0, 2.0 / 3.0], 1.0 / 6.0)],
        Kind::Plane4 | Kind::Axi4 => vec![([-g, 0.0], 1.0), ([g, 0.0], 1.0)],
        _ => vec![([0.0, 0.0], 1.0)],
    }
}

/// Geometri i en integrationspunkt (mittytans nod).
struct Point {
    w: f64,
    n: [f64; 3],
    e2: [f64; 3],
    e3: [f64; 3],
}

/// Elementets geometri: punkterna, membranets B-matris (rader ε₂₂, ε₃₃, ε₂₃
/// mot mittytans nodförskjutningar, 3 komponenter per mittnod) och total area.
struct Geometry {
    pts: Vec<Point>,
    bm: [Vec<f64>; 3],
    area: f64,
    t0: f64,
}

impl Gasket {
    fn dim(&self) -> usize {
        match self.kind {
            Kind::Hex8 | Kind::Wedge6 | Kind::Link3D => 3,
            _ => 2,
        }
    }

    fn half(&self) -> usize {
        match self.kind {
            Kind::Hex8 => 4,
            Kind::Wedge6 => 3,
            Kind::Plane4 | Kind::Axi4 => 2,
            _ => 1,
        }
    }

    fn axi(&self) -> bool {
        matches!(self.kind, Kind::Axi4 | Kind::LinkAxi)
    }

    fn link(&self) -> bool {
        matches!(self.kind, Kind::Link2D | Kind::LinkAxi | Kind::Link3D)
    }

    fn geometry(&self, ctx: &ElemCtx) -> Result<Geometry> {
        let h = self.half();
        let x = &ctx.x;
        let mid: Vec<[f64; 3]> = (0..h).map(|a| [0, 1, 2].map(|i| 0.5 * (x[a][i] + x[a + h][i]))).collect();
        let t_nodes: f64 = (0..h).map(|a| dot(sub(x[a + h], x[a]), sub(x[a + h], x[a])).sqrt()).sum::<f64>() / h as f64;
        let sec = ctx.section(self.name)?;
        let t0 = match sec.raw.first_floats()?.first() {
            Some(&t) if t > 0.0 => t,
            Some(_) => t_nodes,
            None => 1.0,
        };
        let z = [0.0, 0.0, 1.0];
        // tangenter i en naturlig punkt
        let tangents = |p: [f64; 2]| -> ([f64; 3], [f64; 3], Vec<f64>, Vec<[f64; 2]>) {
            let (n, d) = surf_shape(self.kind, p);
            let mut t1 = [0.0; 3];
            let mut t2 = [0.0; 3];
            for a in 0..h {
                for i in 0..3 {
                    t1[i] += d[a][0] * mid[a][i];
                    t2[i] += d[a][1] * mid[a][i];
                }
            }
            (t1, t2, n, d)
        };
        let frame = |t1: [f64; 3], t2: [f64; 3]| -> Result<([f64; 3], [f64; 3], [f64; 3])> {
            let n = match self.kind {
                Kind::Hex8 | Kind::Wedge6 => unit(cross(t1, t2))?,
                Kind::Plane4 | Kind::Axi4 => unit(cross(z, t1))?,
                _ => unit(sub(x[1], x[0]))?,
            };
            if self.dim() == 2 {
                let e2 = cross(z, n);
                return Ok((n, e2, z));
            }
            let gx = [1.0, 0.0, 0.0];
            let mut e2 = sub(gx, [n[0] * n[0], n[0] * n[1], n[0] * n[2]]);
            if dot(e2, e2) < 1e-12 {
                let gz = [0.0, 0.0, 1.0];
                e2 = sub(gz, [n[2] * n[0], n[2] * n[1], n[2] * n[2]]);
            }
            let e2 = unit(e2)?;
            Ok((n, e2, cross(n, e2)))
        };
        let width = 1.0;
        // vikter ∫Nₐ dA
        let mut w = vec![0.0; h];
        if self.link() {
            w[0] = if self.axi() { 2.0 * std::f64::consts::PI * mid[0][0] } else { 1.0 };
        } else {
            for (p, wg) in surf_rule(self.kind) {
                let (t1, t2, n, _) = tangents(p);
                let ja = match self.kind {
                    Kind::Hex8 | Kind::Wedge6 => dot(cross(t1, t2), cross(t1, t2)).sqrt(),
                    _ => dot(t1, t1).sqrt() * width,
                };
                let r: f64 = if self.axi() { 2.0 * std::f64::consts::PI * (0..h).map(|a| n[a] * mid[a][0]).sum::<f64>() } else { 1.0 };
                for a in 0..h {
                    w[a] += wg * ja * r * n[a];
                }
            }
        }
        let (nodes_nat, centre) = surf_nodes(self.kind);
        let mut pts = Vec::with_capacity(h);
        for (a, p) in nodes_nat.iter().enumerate() {
            let (t1, t2, _, _) = tangents(*p);
            let (n, e2, e3) = frame(t1, t2)?;
            pts.push(Point { w: w[a], n, e2, e3 });
        }
        // membran i mitten
        let mut bm = [vec![0.0; 3 * h], vec![0.0; 3 * h], vec![0.0; 3 * h]];
        let area: f64 = w.iter().sum();
        if !self.link() {
            let (t1, t2, nc, d) = tangents(centre);
            let (_n, e2, e3) = frame(t1, t2)?;
            // derivator med avseende på båglängd längs e₂ och e₃
            let (g2, g3): (Vec<f64>, Vec<f64>) = if self.dim() == 3 {
                // [t1 t2] = Jacobian i (e2, e3)-planet
                let (a11, a12, a21, a22) = (dot(t1, e2), dot(t1, e3), dot(t2, e2), dot(t2, e3));
                let det = a11 * a22 - a12 * a21;
                let g2 = (0..h).map(|a| (a22 * d[a][0] - a12 * d[a][1]) / det).collect();
                let g3 = (0..h).map(|a| (-a21 * d[a][0] + a11 * d[a][1]) / det).collect();
                (g2, g3)
            } else {
                let s = dot(t1, e2);
                ((0..h).map(|a| d[a][0] / s).collect(), vec![0.0; h])
            };
            let r: f64 = (0..h).map(|a| nc[a] * mid[a][0]).sum();
            for a in 0..h {
                for i in 0..3 {
                    bm[0][3 * a + i] = g2[a] * e2[i];
                    if self.dim() == 3 {
                        bm[1][3 * a + i] = g3[a] * e3[i];
                        bm[2][3 * a + i] = g3[a] * e2[i] + g2[a] * e3[i];
                    }
                }
                if self.axi() {
                    bm[1][3 * a] = nc[a] / r;
                }
            }
        }
        Ok(Geometry { pts, bm, area, t0 })
    }

    fn ndof_node(&self) -> usize {
        self.dim()
    }

    /// Töjningsmatris (6 rader) för integrationspunkt a mot elementets frihetsgrader.
    fn bmatrix(&self, geo: &Geometry, a: usize) -> Vec<[f64; 6]> {
        let h = self.half();
        let nd = self.ndof_node();
        let mut b = vec![[0.0; 6]; 2 * h * nd];
        let p = &geo.pts[a];
        for i in 0..nd {
            // över (a + h) minus under (a)
            b[(a + h) * nd + i][0] = -p.n[i];
            b[a * nd + i][0] = p.n[i];
            if !self.thickness_only && !self.link() {
                b[(a + h) * nd + i][3] = p.e2[i];
                b[a * nd + i][3] = -p.e2[i];
                if nd == 3 {
                    b[(a + h) * nd + i][4] = p.e3[i];
                    b[a * nd + i][4] = -p.e3[i];
                }
            }
        }
        if !self.thickness_only && !self.link() {
            for c in 0..h {
                for i in 0..nd {
                    for (row, k) in [(1usize, 0usize), (2, 1), (5, 2)] {
                        let v = 0.5 * geo.bm[k][3 * c + i];
                        b[c * nd + i][row] += v;
                        b[(c + h) * nd + i][row] += v;
                    }
                }
            }
        }
        b
    }

    fn evaluate(&self, ctx: &ElemCtx, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<(Vec<f64>, Option<Vec<f64>>, Vec<[f64; 6]>, Vec<[f64; 6]>)> {
        let mat = ctx.material(self.name)?;
        if mat.name() != crate::materials::gasket::NAME {
            return fail(format!("{}: packningselement kräver *GASKET SECTION med BEHAVIOR=", self.name));
        }
        let geo = self.geometry(ctx)?;
        let n = u.len();
        let mut f = vec![0.0; n];
        let mut k = tangent.then(|| vec![0.0; n * n]);
        let stride = 7;
        if sn.len() != stride * self.half() {
            sn.resize(stride * self.half(), 0.0);
        }
        let scale = [1.0, geo.t0, geo.t0, 1.0, 1.0, geo.t0];
        let mut eps_out = Vec::new();
        let mut sig_out = Vec::new();
        for a in 0..self.half() {
            let b = self.bmatrix(&geo, a);
            let mut eps = [0.0; 6];
            for (j, bj) in b.iter().enumerate() {
                for r in 0..6 {
                    eps[r] += bj[r] * u[j];
                }
            }
            let old = &so[a * stride..(a + 1) * stride];
            let deps: Vec<f64> = (0..6).map(|r| eps[r] - 0.0 * old[r]).collect();
            let mut st = [0.0; 1];
            let out = mat.update(
                &MatInput { mode: StressMode::ThreeD, strain: &eps, dstrain: &deps, stress_old: &old[..6], time: ctx.time, dt: ctx.dt, temperature: 0.0, dtemp: 0.0, defgrad: None },
                &old[6..7],
                &mut st,
            )?;
            let w = geo.pts[a].w;
            for (j, bj) in b.iter().enumerate() {
                for r in 0..6 {
                    f[j] += w * bj[r] * out.stress[r] * scale[r];
                }
            }
            if let Some(k) = k.as_mut() {
                for (i, bi) in b.iter().enumerate() {
                    for (j, bj) in b.iter().enumerate() {
                        let mut s = 0.0;
                        for r in 0..6 {
                            if bi[r] == 0.0 {
                                continue;
                            }
                            for c in 0..6 {
                                s += bi[r] * scale[r] * out.tangent[r * 6 + c] * bj[c];
                            }
                        }
                        k[i * n + j] += w * s;
                    }
                }
            }
            let mut sv = [0.0; 6];
            sv.copy_from_slice(&out.stress[..6]);
            sn[a * stride..a * stride + 6].copy_from_slice(&sv);
            sn[a * stride + 6] = st[0];
            eps_out.push(eps);
            sig_out.push(sv);
        }
        let _ = geo.area;
        Ok((f, k, eps_out, sig_out))
    }
}

impl ElementType for Gasket {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        Gasket::dim(self)
    }
    fn n_nodes(&self) -> usize {
        2 * self.half()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if Gasket::dim(self) == 3 {
            &[U1, U2, U3]
        } else {
            &[U1, U2]
        }
    }
    fn n_ips(&self) -> usize {
        self.half()
    }
    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        ctx.material(self.name)?;
        Ok(vec![0.0; 7 * self.half()])
    }
    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return fail(format!("{}: NLGEOM stöds inte ännu för packningselement", self.name));
        }
        let (fint, k, _, _) = self.evaluate(ctx, u, so, sn, tangent)?;
        Ok(ElemResponse { fint, k })
    }
    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        // spänningen ur det godkända tillståndet, töjningen ur förskjutningen
        let geo = self.geometry(ctx)?;
        let three = Gasket::dim(self) == 3;
        let (sl, el): (&[&str], &[&str]) =
            if three { (&["S11", "S22", "S33", "S12", "S13", "S23"], &["E11", "E22", "E33", "E12", "E13", "E23"]) } else { (&["S11", "S22", "S33", "S12"], &["E11", "E22", "E33", "E12"]) };
        let mut s = IpField::new("S", sl);
        let mut e = IpField::new("E", el);
        for a in 0..self.half() {
            let b = self.bmatrix(&geo, a);
            let mut eps = [0.0; 6];
            for (j, bj) in b.iter().enumerate() {
                for r in 0..6 {
                    eps[r] += bj[r] * u[j];
                }
            }
            let sv = state.get(a * 7..a * 7 + 6).map(|x| x.to_vec()).unwrap_or(vec![0.0; 6]);
            let pick = |v: &[f64]| -> Vec<f64> { if three { v[..6].to_vec() } else { vec![v[0], v[1], v[2], v[3]] } };
            s.points.push((a + 1, 0, pick(&sv)));
            e.points.push((a + 1, 0, pick(&eps)));
        }
        Ok(vec![s, e])
    }
}
