//! Område acoustics2: akustiska infinita element (ACIN*) i de akustiska
//! stegen (inkluderas från `acoustic.rs`). Elementmatriserna kommer från
//! `elements/acoustic_infinite.rs`; här assembleras de per material:
//!
//! - *FREQUENCY: styvheten K_g/ρ och massan, utan dämpning (se
//!   `infinite_stiffness`),
//! - *STEADY STATE DYNAMICS: K_g/ρ̃ − i (k̃/ρ̃) C_g − (Ω²/K̃) M_g med komplex
//!   densitet och kompressionsmodul,
//! - *DYNAMIC: styvhet K_g/ρ, dämpning −C_g/(ρc) + γ/(ρK) H_g och massa M_g/K.

use super::*;
use crate::elements::acoustic_infinite as acin;

/// Matriser för de infinita elementen med samma material.
pub(super) struct InfGroup {
    pub props: AcousticProps,
    /// K_g/ρ (alla frihetsgrader).
    pub k: Gen,
    /// C_g/(ρ c).
    pub c: Gen,
    /// M_g/K.
    pub m: Gen,
    /// H_g γ/(ρK) (volymetriskt motstånd i tidsdomänen).
    pub cd: Gen,
}

impl<'a, 'm> Sys<'a, 'm> {
    /// Bygger de infinita elementens matriser och lägger massan i M_f.
    pub(super) fn build_infinite(&mut self) -> Result<Vec<InfGroup>> {
        let m = self.p.model;
        let na = self.n_all();
        let mut groups: Vec<(Option<usize>, InfGroup)> = Vec::new();
        for (e, el) in m.elements.iter().enumerate() {
            let Some(a) = acin::lookup(el.etype.name()) else { continue };
            let props = material_props(m, e)?;
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&nd| m.nodes[nd]).collect();
            let mt = a.matrices(&x, thickness(m, e)).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
            let mi = el.section.and_then(|s| m.sections[s].material_index);
            let gi = match groups.iter().position(|g| g.0 == mi) {
                Some(i) => i,
                None => {
                    groups.push((mi, InfGroup { props: props.clone(), k: Gen::new(na), c: Gen::new(na), m: Gen::new(na), cd: Gen::new(na) }));
                    groups.len() - 1
                }
            };
            let rho = props.density;
            let bulk = props.bulk;
            let c0 = (bulk / rho).sqrt();
            let gamma = props.drag_at(None);
            let s = |v: &[f64], f: f64| -> Vec<f64> { v.iter().map(|x| x * f).collect() };
            let g = &mut groups[gi].1;
            let mut kk = std::mem::take(&mut g.k);
            self.scatter(e, &s(&mt.kg, 1.0 / rho), &mut kk);
            g.k = kk;
            let mut cc = std::mem::take(&mut g.c);
            self.scatter(e, &s(&mt.cg, 1.0 / (rho * c0)), &mut cc);
            g.c = cc;
            let mm = s(&mt.mg, 1.0 / bulk);
            let mut gm = std::mem::take(&mut g.m);
            self.scatter(e, &mm, &mut gm);
            g.m = gm;
            if gamma != 0.0 {
                let mut cd = std::mem::take(&mut g.cd);
                self.scatter(e, &s(&mt.hg, gamma / (rho * bulk)), &mut cd);
                g.cd = cd;
            }
            let mut mf = std::mem::take(&mut self.mf);
            self.scatter(e, &mm, &mut mf);
            self.mf = mf;
        }
        let mut out = Vec::new();
        for (_, mut g) in groups {
            for x in [&mut g.k, &mut g.c, &mut g.m, &mut g.cd] {
                x.finish();
            }
            out.push(g);
        }
        Ok(out)
    }

    /// Styvheten K_g/ρ för alla grupper (för *FREQUENCY). Den osymmetriska
    /// matrisen används (kontrollerat i
    /// tests/cases/acoustics2/acin2d2_cylinder_freq; ½(K + Kᵀ) ger andra värden).
    pub(super) fn infinite_stiffness(&self) -> Gen {
        let mut g = Gen::new(self.n_all());
        for grp in &self.inf {
            for (r, row) in grp.k.rows.iter().enumerate() {
                for &(c, v) in row {
                    g.add(r, c as usize, v);
                }
            }
        }
        g.finish();
        g
    }
}

impl InfGroup {
    /// Koefficienterna (för K, C, M) i frekvensdomänen vid Ω: K_g/ρ̃ =
    /// (K_g/ρ)·(ρ/ρ̃), −i k̃/ρ̃ C_g = (C_g/(ρc))·(−i k̃ ρ c/ρ̃) och
    /// −Ω² (K/K̃ − 1) M_g/K (korrektionen; M_g/K finns redan i M_f).
    pub(super) fn coeffs(&self, w: f64) -> (C, C, C) {
        let p = &self.props;
        let f = w / TWO_PI;
        let (rt, kt) = p.complex_at(f);
        let rho_t = C(rt.0, rt.1);
        let k_t = C(kt.0, kt.1);
        let ck = if w > 0.0 { C(p.density, 0.0).div(rho_t) } else { C(1.0, 0.0) };
        let kw = rho_t.div(k_t).sqrt().scale(w);
        let rc = p.density * (p.bulk / p.density).sqrt();
        let a = kw.div(rho_t).scale(rc);
        let cc = C(a.1, -a.0);
        let q = C(p.bulk, 0.0).div(k_t);
        let cm = C(-w * w * (q.0 - 1.0), -w * w * q.1);
        (ck, cc, cm)
    }
}
