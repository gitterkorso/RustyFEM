//! Area gaps1: *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL for
//! the Timoshenko beams (B21, B22, B31, B32; included from
//! `beam_timoshenko.rs`), geometrically linear.
//!
//! The section forces are uncoupled functions of the generalized strains:
//!
//! ```text
//! N = N(ε),  M1 = M1(κ1),  M2 = M2(κ2),  T = T(φ)
//! ```
//!
//! given with *AXIAL, *M1, *M2 and *TORQUE (`inp/keywords/gaps1.rs`):
//!
//! - LINEAR: the stiffness (one line).
//! - ELASTIC: nonlinear elastic, table from the most negative to the most positive
//!   value through the origin; constant force outside the table.
//! - default (elastic-plastic): table starting at (0, 0) with positive
//!   values; the slope k₀ of the first segment is the elasticity, after that isotropic
//!   hardening (or softening) about the yield limit F_y(ξ) with the accumulated
//!   plastic deformation ξ = Σ|Δx_p|, x_p = x − F/k₀ at the points;
//!   constant outside the table. The same in tension and compression. State per
//!   integration point and relation: (x_p, ξ).
//!
//! A, I11, I12, I22 and J on the data line are used only for the transverse shear:
//! E = k_N/A from the initial slope of the axial relation, G = E/2 (ν = 0) and
//! the shear stiffness G·A with slenderness compensation.
//! If a relation is missing, the stiffness is zero in that direction.

use super::*;
use crate::inp::keywords::gaps1::NLG_PARAM;

/// A section relation.
#[derive(Debug, Clone)]
enum Law {
    None,
    Linear(f64),
    Elastic(Vec<(f64, f64)>),
    /// k₀ and the yield curve (ξ, F_y).
    Plastic(f64, Vec<(f64, f64)>),
}

/// Values per relation and integration point in the state.
const PER: usize = 2;

impl Law {
    fn parse(text: Option<&str>, what: &str) -> Result<Law> {
        let Some(t) = text else { return Ok(Law::None) };
        let mut it = t.split(';');
        let kind = it.next().unwrap_or("");
        let pts: Vec<(f64, f64)> = it
            .filter_map(|p| {
                let mut f = p.split(',');
                Some((f.next()?.parse().ok()?, f.next()?.parse().ok()?))
            })
            .collect();
        match kind {
            "LINEAR" => Ok(Law::Linear(pts.first().map_or(0.0, |p| p.0))),
            "ELASTIC" => {
                // (force, strain) → (strain, force), ascending strain
                let mut v: Vec<(f64, f64)> = pts.iter().map(|&(f, x)| (x, f)).collect();
                v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                if v.len() < 2 {
                    return fail(format!("*{what}, ELASTIC: give at least two points"));
                }
                Ok(Law::Elastic(v))
            }
            _ => {
                let v: Vec<(f64, f64)> = pts.iter().map(|&(f, x)| (x, f)).collect();
                if v.len() < 2 || v[0].0 != 0.0 || v[0].1 != 0.0 || !(v[1].0 > 0.0) {
                    return fail(format!("*{what} (elastic-plastic): the table must start at (0, 0) and continue with positive values"));
                }
                let k0 = v[1].1 / v[1].0;
                let curve: Vec<(f64, f64)> = v[1..].iter().map(|&(x, f)| ((x - f / k0).max(0.0), f)).collect();
                Ok(Law::Plastic(k0, curve))
            }
        }
    }

    /// Initial stiffness.
    fn k0(&self) -> f64 {
        match self {
            Law::None => 0.0,
            Law::Linear(k) => *k,
            Law::Elastic(v) => {
                let i = v.windows(2).position(|w| w[0].0 <= 0.0 && 0.0 <= w[1].0).unwrap_or(0);
                (v[i + 1].1 - v[i].1) / (v[i + 1].0 - v[i].0)
            }
            Law::Plastic(k, _) => *k,
        }
    }

    /// Force and tangent for the strain `x`; `st` = (x_p, ξ) is updated.
    fn eval(&self, x: f64, st_old: &[f64], st_new: &mut [f64]) -> (f64, f64) {
        match self {
            Law::None => (0.0, 0.0),
            Law::Linear(k) => (k * x, *k),
            Law::Elastic(v) => {
                let n = v.len();
                if x <= v[0].0 {
                    return (v[0].1, 0.0);
                }
                if x >= v[n - 1].0 {
                    return (v[n - 1].1, 0.0);
                }
                let i = v.windows(2).position(|w| x >= w[0].0 && x <= w[1].0).unwrap_or(0);
                let s = (v[i + 1].1 - v[i].1) / (v[i + 1].0 - v[i].0);
                (v[i].1 + s * (x - v[i].0), s)
            }
            Law::Plastic(k0, curve) => {
                let (xp, xi) = (st_old[0], st_old[1]);
                let ftr = k0 * (x - xp);
                let (fy, _) = flow(curve, xi);
                if ftr.abs() <= fy {
                    st_new[0] = xp;
                    st_new[1] = xi;
                    return (ftr, *k0);
                }
                // |F_tr| − k₀ Δλ = F_y(ξ + Δλ): the yield curve is piecewise linear
                let mut dl = 0.0;
                for _ in 0..50 {
                    let (f, h) = flow(curve, xi + dl);
                    let r = ftr.abs() - k0 * dl - f;
                    let d = k0 + h;
                    if d.abs() < 1e-300 {
                        break;
                    }
                    let step = r / d;
                    dl += step;
                    if step.abs() <= 1e-14 * (1.0 + dl.abs()) {
                        break;
                    }
                }
                let sgn = ftr.signum();
                st_new[0] = xp + sgn * dl;
                st_new[1] = xi + dl;
                let (f, h) = flow(curve, xi + dl);
                (sgn * f, k0 * h / (k0 + h))
            }
        }
    }
}

/// The yield limit F_y(ξ) and the slope (constant outside the table).
fn flow(c: &[(f64, f64)], xi: f64) -> (f64, f64) {
    let n = c.len();
    if n == 1 || xi <= c[0].0 {
        return (c[0].1, if n > 1 && xi >= c[0].0 { (c[1].1 - c[0].1) / (c[1].0 - c[0].0).max(1e-300) } else { 0.0 });
    }
    if xi >= c[n - 1].0 {
        return (c[n - 1].1, 0.0);
    }
    let i = c.windows(2).position(|w| xi >= w[0].0 && xi < w[1].0).unwrap_or(n - 2);
    let s = (c[i + 1].1 - c[i].1) / (c[i + 1].0 - c[i].0);
    (c[i].1 + s * (xi - c[i].0), s)
}

/// The relations for a NONLINEAR GENERAL section.
pub(super) struct Nlg {
    laws: [Law; 4],
}

/// The element's section is NONLINEAR GENERAL.
pub(super) fn is_nlg(ctx: &ElemCtx) -> bool {
    ctx.section.is_some_and(|s| s.raw.name == "BEAM GENERAL SECTION" && s.raw.param_upper("SECTION").as_deref() == Some("NONLINEAR GENERAL"))
}

impl Nlg {
    pub(super) fn new(ctx: &ElemCtx) -> Result<Nlg> {
        let raw = &ctx.section.unwrap().raw;
        let get = |n: &str| raw.params.iter().find(|(k, _)| k == &format!("{NLG_PARAM}{n}")).map(|(_, v)| v.as_str());
        Ok(Nlg { laws: [Law::parse(get("AXIAL"), "AXIAL")?, Law::parse(get("M1"), "M1")?, Law::parse(get("M2"), "M2")?, Law::parse(get("TORQUE"), "TORQUE")?] })
    }

    fn state_len(ng: usize) -> usize {
        ng * 4 * PER
    }
}

impl TimoBeam {
    /// Geometry with E and G from the initial slopes of the relations.
    fn nlg_geom(&self, ctx: &ElemCtx, nlg: &Nlg) -> Result<BeamGeom> {
        let mut g = self.geom(ctx)?;
        let s = &mut g.sec;
        s.e = if s.a > 0.0 { nlg.laws[0].k0() / s.a } else { 0.0 };
        // the transverse shear with G = E/2 (ν = 0), E from the initial slope of the axial relation
        s.g = s.e / 2.0;
        s.k13 = s.g * s.a;
        s.k23 = s.g * s.a;
        Ok(g)
    }

    /// Generalized strains at the integration points.
    fn nlg_strains(&self, g: &BeamGeom, u: &[f64]) -> Vec<[f64; 6]> {
        let t = self.transform(g);
        let ul = common::matvec(&t, &self.expand_vector(u));
        let nd = 6 * self.nn;
        gauss(self.n_gauss())
            .iter()
            .map(|&(xi, _)| {
                let (b, _) = self.bmat(g, xi);
                std::array::from_fn(|r| (0..nd).map(|k| b[r * nd + k] * ul[k]).sum())
            })
            .collect()
    }

    /// Section forces and tangent for a point (the state is updated).
    fn nlg_point(&self, g: &BeamGeom, nlg: &Nlg, e: &[f64; 6], so: &[f64], sn: &mut [f64]) -> ([f64; 6], [[f64; 6]; 6]) {
        let (k1, k2) = self.shear_stiffness(g);
        let mut s = [0.0; 6];
        let mut c = [[0.0; 6]; 6];
        for (i, law) in nlg.laws.iter().enumerate() {
            let (f, k) = law.eval(e[i], &so[i * PER..(i + 1) * PER], &mut sn[i * PER..(i + 1) * PER]);
            s[i] = f;
            c[i][i] = k;
        }
        s[4] = k1 * e[4];
        s[5] = k2 * e[5];
        c[4][4] = k1;
        c[5][5] = k2;
        (s, c)
    }

    pub(super) fn nlg_init_state(&self) -> Vec<f64> {
        vec![0.0; Nlg::state_len(self.n_gauss())]
    }

    /// Internal force and tangent (geometrically linear).
    pub(super) fn nlg_response(&self, ctx: &ElemCtx, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("NLGEOM with *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL", self.name);
        }
        let nlg = Nlg::new(ctx)?;
        let g = self.nlg_geom(ctx, &nlg)?;
        let ng = self.n_gauss();
        let len = 4 * PER;
        let init;
        let so = if so.len() == ng * len {
            so
        } else {
            init = self.nlg_init_state();
            &init[..]
        };
        sn.clear();
        sn.extend_from_slice(so);
        let t = self.transform(&g);
        let nd = 6 * self.nn;
        let strains = self.nlg_strains(&g, u);
        let mut fl = vec![0.0; nd];
        let mut kl = vec![0.0; nd * nd];
        for (q, &(xi, w)) in gauss(ng).iter().enumerate() {
            let (b, j) = self.bmat(&g, xi);
            let (s, ct) = self.nlg_point(&g, &nlg, &strains[q], &so[q * len..(q + 1) * len], &mut sn[q * len..(q + 1) * len]);
            let wj = w * j;
            for p in 0..nd {
                fl[p] += wj * (0..6).map(|r| b[r * nd + p] * s[r]).sum::<f64>();
            }
            if tangent {
                for r in 0..6 {
                    let crr = ct[r][r];
                    if crr == 0.0 {
                        continue;
                    }
                    for p in 0..nd {
                        let brp = b[r * nd + p];
                        if brp != 0.0 {
                            for k in 0..nd {
                                kl[p * nd + k] += wj * brp * crr * b[r * nd + k];
                            }
                        }
                    }
                }
            }
        }
        let fg: Vec<f64> = (0..nd).map(|jj| (0..nd).map(|i| t[i * nd + jj] * fl[i]).sum()).collect();
        let k = tangent.then(|| self.reduce_matrix(&common::congruence(&kl, &t, nd)));
        Ok(ElemResponse { fint: self.reduce_vector(&fg), k })
    }

    /// Elastic stiffness (the initial slopes) for linear steps.
    pub(super) fn nlg_stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let nlg = Nlg::new(ctx)?;
        let z = vec![0.0; self.n_nodes() * self.node_dofs(0).len()];
        let mut sn = Vec::new();
        let _ = &nlg;
        let r = self.nlg_response(ctx, &z, &self.nlg_init_state(), &mut sn, true)?;
        Ok(r.k.unwrap_or_default())
    }

    /// Section forces from the state (for output).
    pub(super) fn nlg_forces(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<[f64; 6]>> {
        let nlg = Nlg::new(ctx)?;
        let g = self.nlg_geom(ctx, &nlg)?;
        let ng = self.n_gauss();
        let len = 4 * PER;
        let st: Vec<f64> = if state.len() >= ng * len { state[..ng * len].to_vec() } else { self.nlg_init_state() };
        let strains = self.nlg_strains(&g, u);
        let mut out = Vec::new();
        for q in 0..ng {
            // evaluate with the stored state (x_p, ξ) as the start: a
            // converged state gives the same force
            let mut tmp = st[q * len..(q + 1) * len].to_vec();
            let (s, _) = self.nlg_point(&g, &nlg, &strains[q], &st[q * len..(q + 1) * len], &mut tmp);
            out.push(s);
        }
        Ok(out)
    }
}
