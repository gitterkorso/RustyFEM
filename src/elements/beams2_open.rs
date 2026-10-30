//! Öppna tunnväggiga balkar med välvning: B31OS och B32OS (område beams2).
//!
//! Frihetsgrad 7 är välvningsamplituden w. Utöver Timoshenkobalkens
//! generaliserade töjningar (`beam_timoshenko.rs`) finns w och bikrökningen
//! χ = w'. Formuleringen kopplar w till vridningen τ = θt' med en styv elastisk
//! bindning: med p = τ − w är tvärsnittets energi per längdenhet
//!
//!   ½ [böjning och normalkraft som B31] + ½ G J τ² + ½ C_p p² + ½ E Γ χ²
//!     + ½ k1 (γ1 − x2s τ − (c2 − x2s) p)² + ½ k2 (γ2 + x1s τ + (c1 − x1s) p)²,
//!
//! där C_p = G (0,8 Ip − J), Ip = I11 + I22 kring tyngdpunkten (c1, c2),
//! (x1s, x2s) skjuvcentrum, Γ välvningskonstanten (`beams2_sections.rs`) och
//! k1, k2 skjuvstyvheterna med slankhetskompensation som för B31/B32. Med fri
//! välvning (w = τ) blir det precis B31/B32. Uttrycket och faktorn 0,8 är
//! kontrollerade med tvärsnittsmatrisen för B31OS med I- och
//! ARBITRARY-tvärsnitt (`tests/beams2.rs`). Integrering som B31/B32 (1 respektive 2
//! gausspunkter); w interpoleras som förskjutningarna.
//!
//! Utdata: SF, SM (SM3 = vridmoment kring skjuvcentrum) och BIMOM = E Γ χ.
//! Massan är balkens (ingen tröghet för välvningen). Den geometriska
//! styvheten (*BUCKLE, vippning) är Timoshenkobalkens från normalkraft, moment
//! och tvärkrafter, utan bidrag från bimomentet. NLGEOM och olinjära material
//! stöds inte.

use crate::elements::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::elements::beam_timoshenko::{self as timo, BeamGeom, TimoBeam};
use crate::elements::common;
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3, WARP};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct OpenBeam {
    name: &'static str,
    base: &'static TimoBeam,
}

pub static B31OS: OpenBeam = OpenBeam { name: "B31OS", base: &timo::B31 };
pub static B32OS: OpenBeam = OpenBeam { name: "B32OS", base: &timo::B32 };

const DOFS: &[Dof] = &[U1, U2, U3, UR1, UR2, UR3, WARP];

/// Antal generaliserade töjningar: ε, κ1, κ2, τ, γ1, γ2, w, χ.
const NE: usize = 8;

/// Tecken på Ω₀ i kopplingen.
const OMEGA_SIGN: f64 = -1.0;

struct OsData {
    g: BeamGeom,
    d: [[f64; NE]; NE],
}

impl OpenBeam {
    fn nn(&self) -> usize {
        self.base.n_nodes()
    }

    fn data(&self, ctx: &ElemCtx) -> Result<OsData> {
        let sec = ctx.section(self.name)?;
        let Some(b) = sec.beam() else {
            return fail(format!("element {} ({}) kräver *BEAM SECTION eller *BEAM GENERAL SECTION", ctx.label, self.name));
        };
        if let Some(m) = ctx.material {
            if !m.is_linear() && b.moduli.is_none() {
                return not_implemented(&format!("olinjärt material ({})", m.name()), self.name);
            }
        }
        let g = self.base.geom(ctx)?;
        let v = super::sections::data_values(&sec.raw, &b.shape)?;
        let (gamma, ip, om0) = super::sections::open_section_full(&b.shape, &v, &g.sec).map_err(|e| crate::model::FemError(format!("{}: {}", sec.raw.location(), e.0)))?;
        let c = self.base.section_matrix(&g);
        let (k1, k2) = self.base.shear_stiffness(&g);
        let s = &g.sec;
        let mut d = [[0.0; NE]; NE];
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] = c[i][j];
            }
        }
        let mut add = |f: f64, a: &[(usize, f64)]| {
            for &(i, x) in a {
                for &(j, y) in a {
                    d[i][j] += f * x * y;
                }
            }
        };
        add(s.g * s.j, &[(3, 1.0)]);
        if gamma <= 1e-9 * ip * ip / s.a {
            // ingen välvning (t.ex. L): frihetsgrad 7 saknar egen styvhet;
            // här en liten frikopplad styvhet så att systemet inte blir singulärt
            add(k1, &[(4, 1.0), (3, -s.x2s)]);
            add(k2, &[(5, 1.0), (3, s.x1s)]);
            add(1e-12 * s.g * ip, &[(6, 1.0)]);
            add(1e-12 * s.e * ip * g.l * g.l, &[(7, 1.0)]);
            return Ok(OsData { g, d });
        }
        add(s.g * (0.8 * ip - s.j), &[(3, 1.0), (6, -1.0)]);
        let (d1, d2) = (s.c1 - s.x1s, s.c2 - s.x2s);
        add(k1, &[(4, 1.0), (3, -s.x2s - d2), (6, d2)]);
        add(k2, &[(5, 1.0), (3, s.x1s + d1), (6, -d1)]);
        add(s.e * gamma, &[(7, 1.0)]);
        // välvningen i origo (Ω₀ ≠ 0 när origo ligger på tvärsnittet): fibertöjningen
        // innehåller −Ω₀ χ, vilket kopplar χ till ε, κ1 och κ2
        let w = om0 * OMEGA_SIGN;
        d[0][7] += -s.e * w * s.a;
        d[1][7] += -s.e * w * s.s1;
        d[2][7] += s.e * w * s.s2;
        d[7][0] = d[0][7];
        d[7][1] = d[1][7];
        d[7][2] = d[2][7];
        d[7][7] += s.e * om0 * om0 * s.a;
        Ok(OsData { g, d })
    }

    /// B-matris (NE × 7nn) i lokala frihetsgrader [u, v1, v2, θt, θ1, θ2, w] per nod.
    fn bmat(&self, g: &BeamGeom, xi: f64) -> (Vec<f64>, f64) {
        let nn = self.nn();
        let (b6, j) = self.base.bmat(g, xi);
        let (n, dn) = timo::shape(nn, xi);
        let (n6, n7) = (6 * nn, 7 * nn);
        let mut b = vec![0.0; NE * n7];
        for r in 0..6 {
            for i in 0..nn {
                for k in 0..6 {
                    b[r * n7 + 7 * i + k] = b6[r * n6 + 6 * i + k];
                }
            }
        }
        for i in 0..nn {
            b[6 * n7 + 7 * i + 6] = n[i];
            b[7 * n7 + 7 * i + 6] = dn[i] / j;
        }
        (b, j)
    }

    fn transform(&self, g: &BeamGeom) -> Vec<f64> {
        let n7 = 7 * self.nn();
        let mut t = vec![0.0; n7 * n7];
        for i in 0..self.nn() {
            for blk in 0..2 {
                let o = 7 * i + 3 * blk;
                for r in 0..3 {
                    for c in 0..3 {
                        t[(o + r) * n7 + o + c] = g.axes[r][c];
                    }
                }
            }
            t[(7 * i + 6) * n7 + 7 * i + 6] = 1.0;
        }
        t
    }

    fn local_stiffness(&self, os: &OsData) -> Vec<f64> {
        let n7 = 7 * self.nn();
        let mut k = vec![0.0; n7 * n7];
        for &(xi, w) in timo::gauss(self.base.n_gauss()) {
            let (b, j) = self.bmat(&os.g, xi);
            let mut db = vec![0.0; NE * n7];
            for r in 0..NE {
                for s in 0..NE {
                    if os.d[r][s] != 0.0 {
                        for q in 0..n7 {
                            db[r * n7 + q] += os.d[r][s] * b[s * n7 + q];
                        }
                    }
                }
            }
            for r in 0..NE {
                for p in 0..n7 {
                    let brp = b[r * n7 + p];
                    if brp != 0.0 {
                        for q in 0..n7 {
                            k[p * n7 + q] += w * j * brp * db[r * n7 + q];
                        }
                    }
                }
            }
        }
        k
    }

    /// Index för de sex vanliga frihetsgraderna per nod i elementvektorn.
    fn std_index(&self) -> Vec<usize> {
        (0..self.nn()).flat_map(|i| (0..6).map(move |k| 7 * i + k)).collect()
    }

    fn reduce(&self, u: &[f64]) -> Vec<f64> {
        self.std_index().iter().map(|&i| u[i]).collect()
    }

    fn expand_matrix(&self, k6: &[f64]) -> Vec<f64> {
        let idx = self.std_index();
        let (n6, n7) = (idx.len(), 7 * self.nn());
        let mut k = vec![0.0; n7 * n7];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                k[i * n7 + j] = k6[a * n6 + b];
            }
        }
        k
    }

    fn expand_vector(&self, f6: &[f64]) -> Vec<f64> {
        let mut f = vec![0.0; 7 * self.nn()];
        for (a, &i) in self.std_index().iter().enumerate() {
            f[i] = f6[a];
        }
        f
    }

    /// Generaliserade snittkrafter D·e per integrationspunkt.
    fn forces(&self, ctx: &ElemCtx, u: &[f64]) -> Result<(OsData, Vec<[f64; NE]>)> {
        let os = self.data(ctx)?;
        let t = self.transform(&os.g);
        let ul = common::matvec(&t, u);
        let n7 = 7 * self.nn();
        let mut out = Vec::new();
        for &(xi, _) in timo::gauss(self.base.n_gauss()) {
            let (b, _) = self.bmat(&os.g, xi);
            let e: Vec<f64> = (0..NE).map(|r| (0..n7).map(|q| b[r * n7 + q] * ul[q]).sum()).collect();
            let f: [f64; NE] = std::array::from_fn(|r| (0..NE).map(|s| os.d[r][s] * e[s]).sum());
            out.push(f);
        }
        Ok((os, out))
    }
}

impl ElementType for OpenBeam {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        self.nn()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        DOFS
    }
    fn n_ips(&self) -> usize {
        self.base.n_gauss()
    }
    fn vtk_type(&self) -> u8 {
        self.base.vtk_type()
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let os = self.data(ctx)?;
        let t = self.transform(&os.g);
        Ok(common::congruence(&self.local_stiffness(&os), &t, 7 * self.nn()))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(self.expand_matrix(&self.base.mass(ctx, kind)?))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        Ok(self.expand_matrix(&self.base.geometric_stiffness(ctx, &self.reduce(u), state)?))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        Ok(self.expand_vector(&self.base.distributed_load(ctx, load, &self.reduce(u))?))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.base.volume_weights(ctx)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let (os, f) = self.forces(ctx, u)?;
        let s = &os.g.sec;
        let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3"]);
        let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
        let mut bi = IpField::new("BIMOM", &["BIMOM"]);
        for (q, v) in f.iter().enumerate() {
            let [n, m1, m2, tq, v1, v2, _, bm] = *v;
            sf.points.push((q + 1, 0, vec![n, v2, v1]));
            sm.points.push((q + 1, 0, vec![m1 - n * s.c2, m2 + n * s.c1, tq + s.x2s * v1 - s.x1s * v2]));
            bi.points.push((q + 1, 0, vec![bm]));
        }
        Ok(vec![sf, sm, bi])
    }
}
