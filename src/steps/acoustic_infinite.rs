//! Area acoustics2: acoustic infinite elements (ACIN*) in the acoustic
//! steps (included from `acoustic.rs`). The element matrices come from
//! `elements/acoustic_infinite.rs`; here they are assembled per material:
//!
//! - *FREQUENCY: the stiffness K_g/ρ and the mass, without damping (see
//!   `infinite_stiffness`),
//! - *STEADY STATE DYNAMICS: K_g/ρ̃ − i (k̃/ρ̃) C_g − (Ω²/K̃) M_g with complex
//!   density and bulk modulus,
//! - *DYNAMIC: stiffness K_g/ρ, damping −C_g/(ρc) + γ/(ρK) H_g and mass M_g/K.

use super::*;
use crate::elements::acoustic_infinite as acin;

/// Matrices for the infinite elements with the same material.
pub(super) struct InfGroup {
    pub props: AcousticProps,
    /// K_g/ρ (all DOFs).
    pub k: Gen,
    /// C_g/(ρ c).
    pub c: Gen,
    /// M_g/K.
    pub m: Gen,
    /// H_g γ/(ρK) (volumetric drag in the time domain).
    pub cd: Gen,
}

impl<'a, 'm> Sys<'a, 'm> {
    /// Builds the infinite elements' matrices and adds the mass to M_f.
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

    /// The stiffness K_g/ρ for all groups (for *FREQUENCY). The unsymmetric
    /// matrix is used (verified in
    /// tests/cases/acoustics2/acin2d2_cylinder_freq; ½(K + Kᵀ) gives other values).
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
    /// The coefficients (for K, C, M) in the frequency domain at Ω: K_g/ρ̃ =
    /// (K_g/ρ)·(ρ/ρ̃), −i k̃/ρ̃ C_g = (C_g/(ρc))·(−i k̃ ρ c/ρ̃) and
    /// −Ω² (K/K̃ − 1) M_g/K (the correction; M_g/K is already in M_f).
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
