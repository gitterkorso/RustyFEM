//! Piezoelectric elements: C3D8E, C3D20E, CPE4E and CPS4E (displacements
//! and electric potential, degree of freedom 9, at each node).
//!
//! Constitutive relations (*PIEZOELECTRIC TYPE=S and *DIELECTRIC):
//!
//! σ = C ε − eᵀ E,   D = e ε + κ E,   E = −∇φ,
//!
//! with e (3×6, columns 11, 22, 33, 12, 13, 23 and engineering shear strain). With
//! TYPE=E the strain coefficients d are given and e = d C. With g = ∇φ and
//! y = [ε; g], [σ; D] = A y with the symmetric matrix
//!
//! A = [[C, eᵀ], [e, −κ]],
//!
//! and the element matrix K = ∫ Bᵀ A B dV (symmetric, indefinite). The rows for
//! the potential are the charge balance ∫∇δφ·D dV, i.e. the charge residual
//! (−∫∇δφ·D dV, the reaction charge RCHG) with the opposite sign; the step type
//! (`steps/piezo.rs`) flips the sign for charges and reactions.
//! Plane stress: ε33 is condensed out with σ33 = 0 in the mechanical and
//! piezoelectric terms (κ is kept); plane strain: ε33 = 0. In
//! the plane E3 = 0. Full integration (2×2, 2×2×2, 3×3×3); C3D8E and CPE4E
//! have B-bar (averaged volumetric strain) like C3D8 and CPE4,
//! C3D20E and CPS4E do not.
//!
//! Output: S, E, EFLX (D), EPG (the electric field E = −∇φ).

use super::api::{ElemCtx, ElementType, Family, IpField, MassKind};
use super::common::Ip;
use super::heat::{gauss_rule, ip_at, shape_dim, shape_nodes, HeatElem, Shape};
use crate::fem::dof::{Dof, EPOT, U1, U2, U3};
use crate::materials::multiphysics::PiezoProps;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub struct PzElem {
    pub name: &'static str,
    pub shape: Shape,
    pub mode: StressMode,
    /// B-bar like C3D8/CPE4 (C3D8E, CPE4E).
    pub bbar: bool,
    geo: HeatElem,
}

pub static C3D8E: PzElem = PzElem { name: "C3D8E", shape: Shape::Hex8, mode: StressMode::ThreeD, bbar: true, geo: HeatElem { name: "C3D8E", shape: Shape::Hex8, axisym: false } };
pub static C3D20E: PzElem = PzElem { name: "C3D20E", shape: Shape::Hex20, mode: StressMode::ThreeD, bbar: false, geo: HeatElem { name: "C3D20E", shape: Shape::Hex20, axisym: false } };
pub static CPE4E: PzElem = PzElem { name: "CPE4E", shape: Shape::Quad4, mode: StressMode::PlaneStrain, bbar: true, geo: HeatElem { name: "CPE4E", shape: Shape::Quad4, axisym: false } };
pub static CPS4E: PzElem = PzElem { name: "CPS4E", shape: Shape::Quad4, mode: StressMode::PlaneStress, bbar: false, geo: HeatElem { name: "CPS4E", shape: Shape::Quad4, axisym: false } };

pub static TYPES: &[&dyn ElementType] = &[&C3D8E, &C3D20E, &CPE4E, &CPS4E];

static ALL: &[&PzElem] = &[&C3D8E, &C3D20E, &CPE4E, &CPS4E];

/// The piezoelectric element `name`.
pub fn lookup(name: &str) -> Option<&'static PzElem> {
    ALL.iter().copied().find(|e| e.name == name)
}

/// Generalized constitutive matrix A (9×9) for y = [ε (6), ∇φ (3)].
pub fn full_matrix(c: &[f64], pz: &PiezoProps) -> [[f64; 9]; 9] {
    let mut e = pz.coef;
    if pz.strain_type {
        // e = d C
        let d = pz.coef;
        for m in 0..3 {
            for j in 0..6 {
                e[m][j] = (0..6).map(|k| d[m][k] * c[k * 6 + j]).sum();
            }
        }
    }
    let mut a = [[0.0; 9]; 9];
    for i in 0..6 {
        for j in 0..6 {
            a[i][j] = c[i * 6 + j];
        }
    }
    for m in 0..3 {
        for j in 0..6 {
            a[6 + m][j] = e[m][j];
            a[j][6 + m] = e[m][j];
        }
        for n in 0..3 {
            a[6 + m][6 + n] = -pz.kappa[m][n];
        }
    }
    a
}

impl PzElem {
    fn dim(&self) -> usize {
        shape_dim(self.shape)
    }
    fn nn(&self) -> usize {
        shape_nodes(self.shape)
    }
    fn ndof_node(&self) -> usize {
        self.dim() + 1
    }

    /// The matrix with ε33 condensed (σ33 = 0) for plane stress. The
    /// mechanical and piezoelectric terms are condensed but not the
    /// dielectric one (κ is kept).
    fn condensed(&self, a: &[[f64; 9]; 9]) -> [[f64; 9]; 9] {
        let mut c = *a;
        if self.mode == StressMode::PlaneStress {
            let p = a[2][2];
            for i in 0..9 {
                for j in 0..9 {
                    if i >= 6 && j >= 6 {
                        continue;
                    }
                    c[i][j] = a[i][j] - a[i][2] * a[2][j] / p;
                }
            }
        }
        c
    }

    /// The components of y that the element uses (indices in the 9-vector) and the
    /// reduced matrix.
    fn reduced(&self, a: &[[f64; 9]; 9]) -> (Vec<usize>, Vec<f64>) {
        let a = self.condensed(a);
        let idx: Vec<usize> = match self.mode {
            StressMode::ThreeD => (0..9).collect(),
            _ => vec![0, 1, 3, 6, 7],
        };
        let m = idx.len();
        let mut r = vec![0.0; m * m];
        for (i, &p) in idx.iter().enumerate() {
            for (j, &q) in idx.iter().enumerate() {
                r[i * m + j] = a[p][q];
            }
        }
        (idx, r)
    }

    pub fn ips(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let t = ctx.section.map_or(1.0, |s| s.solid_thickness());
        gauss_rule(self.shape).into_iter().map(|(p, w)| ip_at(self.shape, &ctx.x, p, w, t, false)).collect()
    }

    /// B (rows according to `idx`, columns node-wise [u…, φ]).
    fn bmat(&self, ip: &Ip, idx: &[usize], avg: &[[f64; 3]]) -> Vec<f64> {
        let nn = self.nn();
        let nd = self.ndof_node();
        let dim = self.dim();
        let ncol = nn * nd;
        let mut b = vec![0.0; idx.len() * ncol];
        // B-bar: the volumetric strain (planar: ε11 + ε22) is averaged over the element
        let nv = if self.mode == StressMode::ThreeD { 3 } else { 2 };
        for (r, &comp) in idx.iter().enumerate() {
            for a in 0..nn {
                let d = ip.dndx[a];
                let c0 = a * nd;
                let row = &mut b[r * ncol..(r + 1) * ncol];
                match comp {
                    0 => row[c0] = d[0],
                    1 => row[c0 + 1] = d[1],
                    2 if dim == 3 => row[c0 + 2] = d[2],
                    3 => {
                        row[c0] = d[1];
                        row[c0 + 1] = d[0];
                    }
                    4 if dim == 3 => {
                        row[c0] = d[2];
                        row[c0 + 2] = d[0];
                    }
                    5 if dim == 3 => {
                        row[c0 + 1] = d[2];
                        row[c0 + 2] = d[1];
                    }
                    6..=8 => row[c0 + dim] = d[comp - 6],
                    _ => {}
                }
                if !avg.is_empty() && comp < nv {
                    for k in 0..nv {
                        row[c0 + k] += (avg[a][k] - d[k]) / nv as f64;
                    }
                }
            }
        }
        b
    }

    /// Generalized constitutive matrix for the element's material.
    pub fn constitutive(ctx: &ElemCtx, def: &crate::fem::model::MaterialDef, name: &str) -> Result<[[f64; 9]; 9]> {
        let mat = ctx.material(name)?;
        let Some(pz) = PiezoProps::from_def(def)? else {
            return fail(format!("element {} ({name}): the material {} has no *DIELECTRIC", ctx.label, def.name));
        };
        let c = mat.tangent_linear(StressMode::ThreeD)?;
        Ok(full_matrix(&c, &pz))
    }

    /// K = ∫ Bᵀ A B dV (degrees of freedom node-wise: u, φ).
    pub fn matrix(&self, ctx: &ElemCtx, a: &[[f64; 9]; 9]) -> Result<Vec<f64>> {
        let a = *a;
        let (idx, ar) = self.reduced(&a);
        let m = idx.len();
        let ncol = self.nn() * self.ndof_node();
        let mut k = vec![0.0; ncol * ncol];
        let ips = self.ips(ctx)?;
        let avg = self.average(&ips);
        for ip in &ips {
            let b = self.bmat(ip, &idx, &avg);
            // AB
            let mut ab = vec![0.0; m * ncol];
            for r in 0..m {
                for q in 0..m {
                    let v = ar[r * m + q];
                    if v == 0.0 {
                        continue;
                    }
                    for c in 0..ncol {
                        ab[r * ncol + c] += v * b[q * ncol + c];
                    }
                }
            }
            for i in 0..ncol {
                for r in 0..m {
                    let bi = b[r * ncol + i];
                    if bi == 0.0 {
                        continue;
                    }
                    for j in 0..ncol {
                        k[i * ncol + j] += ip.w * bi * ab[r * ncol + j];
                    }
                }
            }
        }
        Ok(k)
    }

    /// Output S, E, EFLX and EPG for the element vector `u` (node-wise u, φ).
    pub fn fields(&self, ctx: &ElemCtx, a: &[[f64; 9]; 9], u: &[f64]) -> Result<Vec<IpField>> {
        let dim = self.dim();
        let (sl, el): (&[&str], &[&str]) = if dim == 3 {
            (&["S11", "S22", "S33", "S12", "S13", "S23"], &["E11", "E22", "E33", "E12", "E13", "E23"])
        } else {
            (&["S11", "S22", "S33", "S12"], &["E11", "E22", "E33", "E12"])
        };
        let vl: &[&str] = if dim == 3 { &["EFLX1", "EFLX2", "EFLX3"] } else { &["EFLX1", "EFLX2"] };
        let gl: &[&str] = if dim == 3 { &["EPG1", "EPG2", "EPG3"] } else { &["EPG1", "EPG2"] };
        let mut fs = IpField::new("S", sl);
        let mut fe = IpField::new("E", el);
        let mut fd = IpField::new("EFLX", vl);
        let mut fg = IpField::new("EPG", gl);
        let comps: &[usize] = if dim == 3 { &[0, 1, 2, 3, 4, 5] } else { &[0, 1, 2, 3] };
        let ac = self.condensed(a);
        let ips = self.ips(ctx)?;
        let avg = self.average(&ips);
        for (q, ip) in ips.iter().enumerate() {
            let mut y = self.y_at(ip, u, &avg);
            if self.mode == StressMode::PlaneStress {
                // ε33 from σ33 = 0
                let s: f64 = (0..9).filter(|&j| j != 2).map(|j| a[2][j] * y[j]).sum();
                y[2] = -s / a[2][2];
            }
            let mut sv = [0.0; 9];
            for i in 0..9 {
                sv[i] = (0..9).map(|j| ac[i][j] * y[j]).sum();
            }
            if self.mode == StressMode::PlaneStress {
                sv[2] = 0.0;
                y[2] = 0.0;
            }
            fs.points.push((q + 1, 0, comps.iter().map(|&c| sv[c]).collect()));
            fe.points.push((q + 1, 0, comps.iter().map(|&c| y[c]).collect()));
            fd.points.push((q + 1, 0, (0..dim).map(|c| sv[6 + c]).collect()));
            fg.points.push((q + 1, 0, (0..dim).map(|c| -y[6 + c]).collect()));
        }
        Ok(vec![fs, fe, fd, fg])
    }

    /// The element mean of ∇N per node for B-bar (empty without B-bar).
    fn average(&self, ips: &[Ip]) -> Vec<[f64; 3]> {
        if !self.bbar {
            return Vec::new();
        }
        let vol: f64 = ips.iter().map(|ip| ip.w).sum();
        (0..self.nn()).map(|a| [0, 1, 2].map(|k| ips.iter().map(|ip| ip.dndx[a][k] * ip.w).sum::<f64>() / vol)).collect()
    }

    /// y = B v at a point (9 components, zeros where the element lacks them).
    fn y_at(&self, ip: &Ip, v: &[f64], avg: &[[f64; 3]]) -> [f64; 9] {
        let all: Vec<usize> = (0..9).collect();
        let b = self.bmat(ip, &all, avg);
        let ncol = v.len();
        let mut y = [0.0; 9];
        for r in 0..9 {
            y[r] = (0..ncol).map(|c| b[r * ncol + c] * v[c]).sum();
        }
        y
    }
}

impl ElementType for PzElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        if self.dim() == 3 {
            Family::Continuum3D
        } else {
            Family::Continuum2D
        }
    }
    fn dim(&self) -> usize {
        shape_dim(self.shape)
    }
    fn n_nodes(&self) -> usize {
        shape_nodes(self.shape)
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.dim() == 3 {
            &[U1, U2, U3, EPOT]
        } else {
            &[U1, U2, EPOT]
        }
    }
    fn n_ips(&self) -> usize {
        gauss_rule(self.shape).len()
    }
    fn vtk_type(&self) -> u8 {
        ElementType::vtk_type(&self.geo)
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.geo.faces()
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let nn = self.nn();
        let nd = self.ndof_node();
        let dim = self.dim();
        let ncol = nn * nd;
        let mut m = vec![0.0; ncol * ncol];
        let rho = ctx.density;
        for ip in self.ips(ctx)? {
            for a in 0..nn {
                for b in 0..nn {
                    let v = rho * ip.n[a] * ip.n[b] * ip.w;
                    for c in 0..dim {
                        match kind {
                            MassKind::Consistent => m[(a * nd + c) * ncol + b * nd + c] += v,
                            MassKind::Lumped => m[(a * nd + c) * ncol + a * nd + c] += v,
                        }
                    }
                }
            }
        }
        Ok(m)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(super::common::volume_weights(&self.ips(ctx)?))
    }

}
