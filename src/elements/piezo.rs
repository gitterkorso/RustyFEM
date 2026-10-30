//! Piezoelektriska element: C3D8E, C3D20E, CPE4E och CPS4E (förskjutningar
//! och elektrisk potential, frihetsgrad 9, i varje nod).
//!
//! Konstitutiva samband (*PIEZOELECTRIC TYPE=S och *DIELECTRIC):
//!
//! σ = C ε − eᵀ E,   D = e ε + κ E,   E = −∇φ,
//!
//! med e (3×6, kolumner 11, 22, 33, 12, 13, 23 och ingenjörsskjuvtöjning). Med
//! TYPE=E är töjningskoefficienterna d givna och e = d C. Med g = ∇φ och
//! y = [ε; g] är [σ; D] = A y med den symmetriska matrisen
//!
//! A = [[C, eᵀ], [e, −κ]],
//!
//! och elementmatrisen K = ∫ Bᵀ A B dV (symmetrisk, indefinit). Raderna för
//! potentialen är laddningsbalansen ∫∇δφ·D dV, dvs. laddningsresidualen
//! (−∫∇δφ·D dV, reaktionsladdningen RCHG) med omvänt tecken; steg-typen
//! (`steps/piezo.rs`) vänder tecknet för laddningar och reaktioner.
//! Plan spänning: ε33 kondenseras bort med σ33 = 0 i de mekaniska och
//! piezoelektriska termerna (κ behålls); plan töjning: ε33 = 0. I
//! planet är E3 = 0. Full integrering (2×2, 2×2×2, 3×3×3); C3D8E och CPE4E
//! har B-bar (medelvärdesbildad volymtöjning) som C3D8 och CPE4,
//! C3D20E och CPS4E inte.
//!
//! Utdata: S, E, EFLX (D), EPG (elektriska fältet E = −∇φ).

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
    /// B-bar som C3D8/CPE4 (C3D8E, CPE4E).
    pub bbar: bool,
    geo: HeatElem,
}

pub static C3D8E: PzElem = PzElem { name: "C3D8E", shape: Shape::Hex8, mode: StressMode::ThreeD, bbar: true, geo: HeatElem { name: "C3D8E", shape: Shape::Hex8, axisym: false } };
pub static C3D20E: PzElem = PzElem { name: "C3D20E", shape: Shape::Hex20, mode: StressMode::ThreeD, bbar: false, geo: HeatElem { name: "C3D20E", shape: Shape::Hex20, axisym: false } };
pub static CPE4E: PzElem = PzElem { name: "CPE4E", shape: Shape::Quad4, mode: StressMode::PlaneStrain, bbar: true, geo: HeatElem { name: "CPE4E", shape: Shape::Quad4, axisym: false } };
pub static CPS4E: PzElem = PzElem { name: "CPS4E", shape: Shape::Quad4, mode: StressMode::PlaneStress, bbar: false, geo: HeatElem { name: "CPS4E", shape: Shape::Quad4, axisym: false } };

pub static TYPES: &[&dyn ElementType] = &[&C3D8E, &C3D20E, &CPE4E, &CPS4E];

static ALL: &[&PzElem] = &[&C3D8E, &C3D20E, &CPE4E, &CPS4E];

/// Det piezoelektriska elementet `name`.
pub fn lookup(name: &str) -> Option<&'static PzElem> {
    ALL.iter().copied().find(|e| e.name == name)
}

/// Generaliserad konstitutiv matris A (9×9) för y = [ε (6), ∇φ (3)].
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

    /// Matrisen med ε33 kondenserad (σ33 = 0) för plan spänning. De
    /// mekaniska och piezoelektriska termerna kondenseras men inte den
    /// dielektriska (κ behålls).
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

    /// Komponenterna i y som elementet använder (index i 9-vektorn) och den
    /// reducerade matrisen.
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

    /// B (rader enligt `idx`, kolumner nodvis [u…, φ]).
    fn bmat(&self, ip: &Ip, idx: &[usize], avg: &[[f64; 3]]) -> Vec<f64> {
        let nn = self.nn();
        let nd = self.ndof_node();
        let dim = self.dim();
        let ncol = nn * nd;
        let mut b = vec![0.0; idx.len() * ncol];
        // B-bar: volymtöjningen (plana: ε11 + ε22) medelvärdesbildas över elementet
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

    /// Generaliserad konstitutiv matris för elementets material.
    pub fn constitutive(ctx: &ElemCtx, def: &crate::fem::model::MaterialDef, name: &str) -> Result<[[f64; 9]; 9]> {
        let mat = ctx.material(name)?;
        let Some(pz) = PiezoProps::from_def(def)? else {
            return fail(format!("element {} ({name}): materialet {} saknar *DIELECTRIC", ctx.label, def.name));
        };
        let c = mat.tangent_linear(StressMode::ThreeD)?;
        Ok(full_matrix(&c, &pz))
    }

    /// K = ∫ Bᵀ A B dV (frihetsgraderna nodvis: u, φ).
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

    /// Utdata S, E, EFLX och EPG för elementvektorn `u` (nodvis u, φ).
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
                // ε33 ur σ33 = 0
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

    /// Elementmedelvärdet av ∇N per nod för B-bar (tomt utan B-bar).
    fn average(&self, ips: &[Ip]) -> Vec<[f64; 3]> {
        if !self.bbar {
            return Vec::new();
        }
        let vol: f64 = ips.iter().map(|ip| ip.w).sum();
        (0..self.nn()).map(|a| [0, 1, 2].map(|k| ips.iter().map(|ip| ip.dndx[a][k] * ip.w).sum::<f64>() / vol)).collect()
    }

    /// y = B v i en punkt (9 komponenter, nollor där elementet saknar).
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
