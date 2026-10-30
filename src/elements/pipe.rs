//! Rörelement PIPE21, PIPE22 (plana) och PIPE31, PIPE32 (rumsliga).
//!
//! Rörelementen är Timoshenkobalkarna B21/B22/B31/B32 (`beam_timoshenko.rs`)
//! med tunnväggigt rörtvärsnitt (`*BEAM SECTION, SECTION=PIPE` eller
//! `*BEAM GENERAL SECTION, SECTION=PIPE`: ytterradie r, väggtjocklek t) och
//! en extra variabel per integrationspunkt för ringtöjningen εθ (1 för
//! PIPE21/PIPE31, 2 för PIPE22/PIPE32). Ringtöjningen ligger i
//! U1 på en intern nod per integrationspunkt (`internal_nodes`).
//!
//! Formulering:
//!
//! - Väggens membrandel är i plan spänning med Ē = E/(1 − ν²):
//!   N = Ē·A·(ε + ν·εθ) och ringkraften Nθ = Ē·A·(εθ + ν·ε) per längdenhet.
//!   Böjning, vridning och skjuvning är balkens (E·I, G·J, k·G·A), dvs.
//!   fibrernas ringspänning är densamma i hela tvärsnittet.
//! - Utan tryck blir εθ = −ν·ε och N = E·A·ε (som balken).
//! - *DLOAD PI/PE (inre/yttre tryck p med effektiv diameter D på dataraden,
//!   slutna ändar): ringlasten ger σθ = (pᵢ·Dᵢ − pₑ·Dₑ)/(2t) och
//!   ändlocksbidraget P = π/4·(pᵢ·Dᵢ² − pₑ·Dₑ²) läggs som axialkrafter −P·t
//!   och +P·t i elementets ändnoder (t = elementets axel). Längs ett
//!   sammanhängande rör tar ändlockskrafterna ut varandra, så rörets
//!   normalkraft SF1 = P plus Poissoneffekten men den effektiva normalkraften
//!   (ESF1 = SF1 − P) bestäms av jämvikten – ingen reaktion uppstår av trycket
//!   i ett fast inspänt rör.
//!
//! Utdata: SF/SM som balkarna (SF1 = rörväggens normalkraft) och för
//! `*BEAM SECTION` spänningen S (S11 axiell, S22 ring, S12 vridskjuvning) och
//! töjningen E i standardpunkterna på medelradien: 1, 3, 5, 7 (rumsliga,
//! vid x2 = −rₘ, x1 = rₘ, x2 = rₘ, x1 = −rₘ) respektive 1 och 5 (plana,
//! x2 = −rₘ och rₘ).
//!
//! Med olinjärt material eller NLGEOM räknas balkdelen som B2x/B3x
//! (sektionspunkter respektive den geometriskt olinjära balken) och
//! ringtöjningen är då frikopplad från axialtöjningen (trycket ger inget
//! Poissonbidrag).

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::beam_timoshenko::{self as timo, plastic, BeamGeom, TimoBeam};
use super::common::{self, norm, scale};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct Pipe {
    name: &'static str,
    beam: &'static TimoBeam,
    dim: usize,
    /// Antal balknoder (2 eller 3).
    nb: usize,
}

pub static PIPE21: Pipe = Pipe { name: "PIPE21", beam: &timo::B21, dim: 2, nb: 2 };
pub static PIPE22: Pipe = Pipe { name: "PIPE22", beam: &timo::B22, dim: 2, nb: 3 };
pub static PIPE31: Pipe = Pipe { name: "PIPE31", beam: &timo::B31, dim: 3, nb: 2 };
pub static PIPE32: Pipe = Pipe { name: "PIPE32", beam: &timo::B32, dim: 3, nb: 3 };

pub static TYPES: &[&dyn ElementType] = &[&PIPE21, &PIPE22, &PIPE31, &PIPE32];

/// Gausspunkter som balkarna (1 för linjära, 2 för kvadratiska element).
fn gauss(nb: usize) -> &'static [(f64, f64)] {
    const G1: &[(f64, f64)] = &[(0.0, 2.0)];
    const G2P: &[(f64, f64)] = &[(-common::G2, 1.0), (common::G2, 1.0)];
    if nb == 2 { G1 } else { G2P }
}

fn dshape(nb: usize, xi: f64) -> Vec<f64> {
    if nb == 2 { vec![-0.5, 0.5] } else { vec![xi - 0.5, -2.0 * xi, xi + 0.5] }
}

/// Rörtvärsnittets materialdata.
struct Wall {
    e: f64,
    nu: f64,
    /// Ē = E/(1 − ν²)
    eb: f64,
    a: f64,
    /// Ytterradie och väggtjocklek (0 om de saknas, t.ex. GENERAL).
    r: f64,
    t: f64,
}

impl Pipe {
    fn ndof_beam(&self) -> usize {
        self.nb * if self.dim == 2 { 3 } else { 6 }
    }

    fn nh(&self) -> usize {
        self.nb - 1
    }

    fn tdofs(&self) -> usize {
        if self.dim == 2 { 3 } else { 6 }
    }

    /// Elementets sammanhang för balkdelen (utan de interna noderna).
    fn beam_ctx<'a>(&self, ctx: &ElemCtx<'a>) -> ElemCtx<'a> {
        let mut b = ctx.clone();
        let nn = self.n_nodes();
        b.x.truncate(self.nb);
        let fix = |v: &mut Vec<f64>| {
            if v.len() == nn {
                v.truncate(self.nb);
            } else if v.len() == 2 * nn {
                let g: Vec<f64> = v[nn..nn + self.nb].to_vec();
                v.truncate(self.nb);
                v.extend(g);
            }
        };
        fix(&mut b.temp);
        fix(&mut b.temp_old);
        b
    }

    fn wall(&self, ctx: &ElemCtx, g: &BeamGeom) -> Result<Wall> {
        let sec = ctx.section(self.name)?;
        let shape = sec.beam().map(|b| b.shape.to_uppercase()).unwrap_or_default();
        if shape != "PIPE" {
            return fail(format!(
                "element {} ({}): rörelement kräver SECTION=PIPE på *BEAM SECTION eller *BEAM GENERAL SECTION (inte {shape})",
                ctx.label, self.name
            ));
        }
        let v = sec.raw.first_floats()?;
        let (r, t) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        let e = g.sec.e;
        // *BEAM GENERAL SECTION har inget tvärkontraktionstal: ring- och
        // axialtöjningen kopplas då inte (ν = POISSON=, standard 0)
        let nu = if sec.raw.name == "BEAM GENERAL SECTION" {
            sec.raw.param("POISSON").and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(0.0)
        } else if g.sec.g > 0.0 {
            e / (2.0 * g.sec.g) - 1.0
        } else {
            0.0
        };
        Ok(Wall { e, nu, eb: e / (1.0 - nu * nu), a: g.sec.a, r, t })
    }

    /// Axialtöjningens B-vektor (över balkfrihetsgraderna) och vikt w·|dx/dξ|
    /// i gausspunkt `q`.
    fn axial_b(&self, g: &BeamGeom, q: usize) -> (Vec<f64>, f64) {
        let (xi, w) = gauss(self.nb)[q];
        let dn = dshape(self.nb, xi);
        let mut d = [0.0; 3];
        for i in 0..self.nb {
            for k in 0..3 {
                d[k] += dn[i] * g.x[i][k];
            }
        }
        let j = norm(d);
        let t = g.axes[0];
        let nt = self.tdofs();
        let mut b = vec![0.0; self.ndof_beam()];
        for i in 0..self.nb {
            for c in 0..self.dim {
                b[i * nt + c] = dn[i] / j * t[c];
            }
        }
        (b, w * j)
    }

    /// Lägger in en balkmatris (nb-frihetsgrader) i elementmatrisen.
    fn embed(&self, kb: &[f64]) -> Vec<f64> {
        let nbd = self.ndof_beam();
        let n = nbd + self.nh();
        let mut k = vec![0.0; n * n];
        for i in 0..nbd {
            k[i * n..i * n + nbd].copy_from_slice(&kb[i * nbd..(i + 1) * nbd]);
        }
        k
    }

    fn pad(&self, f: Vec<f64>) -> Vec<f64> {
        let mut f = f;
        f.resize(self.ndof_beam() + self.nh(), 0.0);
        f
    }

    /// Ringtöjningens styvhet (frikopplad) eller membrankopplingen (linjär).
    fn add_hoop(&self, ctx: &ElemCtx, k: &mut [f64], coupled: bool) -> Result<()> {
        let bctx = self.beam_ctx(ctx);
        let g = self.beam.geom(&bctx)?;
        let wl = self.wall(ctx, &g)?;
        let nbd = self.ndof_beam();
        let n = nbd + self.nh();
        for q in 0..self.nh() {
            let (b, wj) = self.axial_b(&g, q);
            let h = nbd + q;
            k[h * n + h] += wl.eb * wl.a * wj;
            if coupled {
                let c = (wl.eb - wl.e) * wl.a * wj;
                let cu = wl.nu * wl.eb * wl.a * wj;
                for i in 0..nbd {
                    if b[i] == 0.0 {
                        continue;
                    }
                    k[i * n + h] += cu * b[i];
                    k[h * n + i] += cu * b[i];
                    for jj in 0..nbd {
                        k[i * n + jj] += c * b[i] * b[jj];
                    }
                }
            }
        }
        Ok(())
    }

    fn linear(&self, ctx: &ElemCtx) -> bool {
        !ctx.nlgeom && !plastic::is_plastic(ctx)
    }
}

impl ElementType for Pipe {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        self.nb + self.nh()
    }
    fn n_connectivity(&self) -> usize {
        self.nb
    }
    fn internal_nodes(&self) -> usize {
        self.nh()
    }
    fn node_dofs(&self, i: usize) -> &'static [Dof] {
        if i >= self.nb {
            &[U1]
        } else if self.dim == 2 {
            &[U1, U2, UR3]
        } else {
            &[U1, U2, U3, UR1, UR2, UR3]
        }
    }
    fn n_ips(&self) -> usize {
        self.nh()
    }
    fn vtk_type(&self) -> u8 {
        self.beam.vtk_type()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.beam.init_state(&self.beam_ctx(ctx))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let kb = self.beam.stiffness(&self.beam_ctx(ctx))?;
        let mut k = self.embed(&kb);
        self.add_hoop(ctx, &mut k, true)?;
        Ok(k)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let nbd = self.ndof_beam();
        let n = nbd + self.nh();
        if self.linear(ctx) {
            let k = self.stiffness(ctx)?;
            return Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) });
        }
        // olinjärt material eller NLGEOM: balken som B2x/B3x, ringtöjningen frikopplad
        let r = self.beam.internal_force(&self.beam_ctx(ctx), &u[..nbd], &du[..nbd], so, sn, tangent)?;
        let mut kh = vec![0.0; n * n];
        self.add_hoop(ctx, &mut kh, false)?;
        let mut fint = self.pad(r.fint);
        for h in nbd..n {
            fint[h] += kh[h * n + h] * u[h];
        }
        let k = r.k.map(|kb| {
            let mut k = self.embed(&kb);
            for h in nbd..n {
                k[h * n + h] += kh[h * n + h];
            }
            k
        });
        Ok(ElemResponse { fint, k })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(self.embed(&self.beam.mass(&self.beam_ctx(ctx), kind)?))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        let nbd = self.ndof_beam();
        Ok(self.embed(&self.beam.geometric_stiffness(&self.beam_ctx(ctx), &u[..nbd], state)?))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let nbd = self.ndof_beam();
        let bctx = self.beam_ctx(ctx);
        let (pi, pe) = match load.kind.as_str() {
            "PI" => (load.magnitude, 0.0),
            "PE" => (0.0, load.magnitude),
            _ => return Ok(self.pad(self.beam.distributed_load(&bctx, load, &u[..nbd])?)),
        };
        let d = load.extra.first().copied().unwrap_or(0.0);
        if !(d > 0.0) {
            return fail(format!(
                "element {} ({}): *DLOAD {} kräver den effektiva diametern efter storleken",
                ctx.label, self.name, load.kind
            ));
        }
        let g = self.beam.geom(&bctx)?;
        let wl = self.wall(ctx, &g)?;
        if !(wl.t > 0.0) {
            return fail(format!("element {} ({}): rörets väggtjocklek saknas", ctx.label, self.name));
        }
        let mut f = vec![0.0; nbd + self.nh()];
        // ändlocken: axialkrafter längs elementets axel i ändnoderna
        let cap = std::f64::consts::PI / 4.0 * (pi - pe) * d * d;
        let tdir = |xi: f64| -> [f64; 3] {
            let dn = dshape(self.nb, xi);
            let mut v = [0.0; 3];
            for i in 0..self.nb {
                for k in 0..3 {
                    v[k] += dn[i] * g.x[i][k];
                }
            }
            scale(v, 1.0 / norm(v))
        };
        let nt = self.tdofs();
        let (t0, t1) = (tdir(-1.0), tdir(1.0));
        for c in 0..self.dim {
            f[c] -= cap * t0[c];
            f[(self.nb - 1) * nt + c] += cap * t1[c];
        }
        // ringlasten: σθ = (pᵢDᵢ − pₑDₑ)/(2t) i jämvikt med Ē·A·(εθ + ν·ε)
        let sth = (pi - pe) * d / (2.0 * wl.t);
        for q in 0..self.nh() {
            let (_, wj) = self.axial_b(&g, q);
            f[nbd + q] += sth * wl.a * wj;
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut v = self.beam.volume_weights(&self.beam_ctx(ctx))?;
        v.resize(self.n_nodes(), 0.0);
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let nbd = self.ndof_beam();
        let bctx = self.beam_ctx(ctx);
        let mut out = self.beam.output(&bctx, &u[..nbd], state)?;
        if !self.linear(ctx) {
            return Ok(out);
        }
        let g = self.beam.geom(&bctx)?;
        let wl = self.wall(ctx, &g)?;
        let mut sf1 = Vec::new();
        let mut sth = Vec::new();
        for q in 0..self.nh() {
            let (b, _) = self.axial_b(&g, q);
            let eps: f64 = b.iter().zip(&u[..nbd]).map(|(a, x)| a * x).sum();
            let eh = u[nbd + q];
            sf1.push(wl.eb * wl.a * (eps + wl.nu * eh));
            sth.push(wl.eb * (eh + wl.nu * eps));
        }
        let mut sm = Vec::new();
        for f in out.iter_mut() {
            if f.name == "SF" {
                for (q, p) in f.points.iter_mut().enumerate() {
                    p.2[0] = sf1[q];
                }
            }
            if f.name == "SM" {
                sm = f.points.iter().map(|p| p.2.clone()).collect();
            }
        }
        // spänningar i standardpunkterna (bara *BEAM SECTION)
        let sec = ctx.section(self.name)?;
        if sec.raw.name != "BEAM SECTION" || wl.t <= 0.0 || sm.len() != self.nh() {
            return Ok(out);
        }
        let rm = wl.r - wl.t / 2.0;
        let s = &g.sec;
        let pts: Vec<(usize, f64, f64)> = if self.dim == 2 {
            vec![(1, 0.0, -rm), (5, 0.0, rm)]
        } else {
            vec![(1, 0.0, -rm), (3, rm, 0.0), (5, 0.0, rm), (7, -rm, 0.0)]
        };
        let mut fs = IpField::new("S", &["S11", "S22", "S12"]);
        let mut fe = IpField::new("E", &["E11", "E22", "E12"]);
        let gm = if wl.nu > -1.0 { wl.e / (2.0 * (1.0 + wl.nu)) } else { wl.e };
        for q in 0..self.nh() {
            let (m1, m2, tq) = (sm[q][0], sm[q][1], sm[q][2]);
            for &(k, x1, x2) in &pts {
                let s11 = sf1[q] / wl.a + if s.i11 > 0.0 { m1 * x2 / s.i11 } else { 0.0 } - if s.i22 > 0.0 { m2 * x1 / s.i22 } else { 0.0 };
                let s12 = if self.dim == 3 && s.j > 0.0 { tq * rm / s.j } else { 0.0 };
                let s22 = sth[q];
                fs.points.push((q + 1, k, vec![s11, s22, s12]));
                fe.points.push((q + 1, k, vec![(s11 - wl.nu * s22) / wl.e, (s22 - wl.nu * s11) / wl.e, s12 / gm]));
            }
        }
        out.push(fs);
        out.push(fe);
        Ok(out)
    }
}
