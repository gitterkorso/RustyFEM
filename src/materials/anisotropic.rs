//! Anisotropic linear elasticity and local material directions.
//!
//! *ELASTIC with TYPE=
//! - ENGINEERING CONSTANTS: E1, E2, E3, ν12, ν13, ν23, G12, G13, G23
//!   (orthotropic, specified via the compliance),
//! - ORTHOTROPIC: D1111, D1122, D2222, D1133, D2233, D3333, D1212, D1313, D2323,
//! - ANISOTROPIC: the 21 constants of the upper triangle column by column
//!   (D1111, D1122, D2222, D1133, D2233, D3333, D1112, ..., D2323),
//! - LAMINA: E1, E2, ν12, G12, G13, G23 (plane stress only).
//!
//! The constitutive relation is σ = D ε in the component order (11, 22, 33,
//! 12, 13, 23) with engineering shear strain, so D1212 = G12 etc. For plane
//! stress and beams/trusses the stress components that must be zero are condensed
//! out (static condensation of the 3D matrix), for plane strain the
//! submatrix for (11, 22, 33, 12) is taken.
//!
//! Local material directions (*ORIENTATION, SYSTEM=RECTANGULAR or
//! CYLINDRICAL, DEFINITION=COORDINATES, with an optional extra rotation about a
//! local axis) are attached to elements via the section parameter ORIENTATION=.
//! The continuum core (`elements/common.rs`) rotates the strain to the local
//! axes, calls the material and rotates the stress and tangent back;
//! stress and strain are reported in local axes. The directions
//! are computed at each integration point (matters for cylindrical systems).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::elements::common::{cross, dot, inv3, norm, scale, sub};
use crate::fem::model::{rotate_vec, FeModel, MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "ELASTIC (anisotropic)", consumes: &["ELASTIC"], priority: 1, build: Some(build) }];

/// Indices of the components of a stress state among the six 3D components.
pub fn mode_components(mode: StressMode) -> &'static [usize] {
    match mode {
        StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
        StressMode::PlaneStrain | StressMode::Axisymmetric => &[0, 1, 2, 3],
        StressMode::PlaneStress => &[0, 1, 3],
        StressMode::Uniaxial => &[0],
    }
}

/// Components that are stress-free (condensed out) in a stress state.
pub fn free_components(mode: StressMode) -> &'static [usize] {
    match mode {
        StressMode::PlaneStress => &[2, 4, 5],
        StressMode::Uniaxial => &[1, 2, 3, 4, 5],
        _ => &[],
    }
}

/// Solves A x = b (n×n row-major) with Gaussian elimination and pivoting.
pub fn solve_dense(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs()))?;
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
            }
            x.swap(c, p);
        }
        for r in c + 1..n {
            let f = m[r * n + c] / m[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    m[r * n + k] -= f * m[c * n + k];
                }
                x[r] -= f * x[c];
            }
        }
    }
    for c in (0..n).rev() {
        let mut s = x[c];
        for k in c + 1..n {
            s -= m[c * n + k] * x[k];
        }
        x[c] = s / m[c * n + c];
    }
    Some(x)
}

/// Inverse of an n×n matrix (row-major).
pub fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = solve_dense(a, &e)?;
        for i in 0..n {
            inv[i * n + j] = col[i];
        }
    }
    Some(inv)
}

/// Reduces a 6×6 stiffness matrix to the stress state `mode`:
/// submatrix for the given components, static condensation of the stress-free ones.
pub fn reduce_6x6(d: &[[f64; 6]; 6], mode: StressMode) -> Vec<f64> {
    let a = mode_components(mode);
    let b = free_components(mode);
    let na = a.len();
    let mut out: Vec<f64> = a.iter().flat_map(|&i| a.iter().map(move |&j| d[i][j])).collect();
    if b.is_empty() {
        return out;
    }
    let nb = b.len();
    let dbb: Vec<f64> = b.iter().flat_map(|&i| b.iter().map(move |&j| d[i][j])).collect();
    let Some(inv) = invert(&dbb, nb) else { return out };
    for (r, &i) in a.iter().enumerate() {
        for (c, &j) in a.iter().enumerate() {
            let mut s = 0.0;
            for p in 0..nb {
                for q in 0..nb {
                    s += d[i][b[p]] * inv[p * nb + q] * d[b[q]][j];
                }
            }
            out[r * na + c] -= s;
        }
    }
    out
}

/// Linear anisotropic elastic material with the 6×6 matrix `d` (local axes),
/// or a ready-made matrix for plane stress (LAMINA).
#[derive(Debug, Clone)]
pub struct AnisotropicElastic {
    pub d: [[f64; 6]; 6],
    /// LAMINA: 3×3 matrix for plane stress (11, 22, 12).
    pub lamina: Option<[f64; 9]>,
    pub kind: String,
}

impl AnisotropicElastic {
    pub fn dmat(&self, mode: StressMode) -> Result<Vec<f64>> {
        if let Some(l) = &self.lamina {
            return match mode {
                StressMode::PlaneStress => Ok(l.to_vec()),
                StressMode::Uniaxial => Ok(vec![l[0] - l[1] * l[3] / l[4]]),
                _ => fail("*ELASTIC, TYPE=LAMINA applies only to plane stress (CPS elements and shells)"),
            };
        }
        Ok(reduce_6x6(&self.d, mode))
    }
}

/// Parses all numbers in an *ELASTIC block (continuation lines are merged).
fn all_floats(kw: &RawKeyword) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    for d in &kw.data {
        v.extend(kw.floats(d)?);
    }
    Ok(v)
}

/// 6×6 stiffness for an orthotropic material from engineering constants.
pub fn engineering_constants(c: &[f64; 9]) -> Result<[[f64; 6]; 6]> {
    let [e1, e2, e3, n12, n13, n23, g12, g13, g23] = *c;
    if !(e1 > 0.0 && e2 > 0.0 && e3 > 0.0 && g12 > 0.0 && g13 > 0.0 && g23 > 0.0) {
        return fail("moduli in ENGINEERING CONSTANTS must be positive");
    }
    let s = [
        [1.0 / e1, -n12 / e1, -n13 / e1],
        [-n12 / e1, 1.0 / e2, -n23 / e2],
        [-n13 / e1, -n23 / e2, 1.0 / e3],
    ];
    let (inv, det) = inv3(s);
    if !(det > 0.0) {
        return fail("ENGINEERING CONSTANTS give a compliance matrix that is not positive definite (check ν)");
    }
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = inv[i][j];
        }
    }
    d[3][3] = g12;
    d[4][4] = g13;
    d[5][5] = g23;
    Ok(d)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("ELASTIC") else { return Ok(None) };
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    let ty = ty.split_whitespace().collect::<Vec<_>>().join(" ");
    if ty == "ISOTROPIC" || ty == "SHEAR" {
        // TYPE=SHEAR (deviatoric part for *EOS) is handled by `materials/eos.rs`
        return Ok(None);
    }
    if ty == "TRACTION" {
        return Ok(None); // cohesive elements: materials/cohesive.rs
    }
    if kw.has("DEPENDENCIES") || kw.has("TEMPERATURE") {
        return kw.err("temperature-dependent elasticity is not supported yet");
    }
    let v = all_floats(kw)?;
    let need = |n: usize| -> Result<()> {
        if v.len() < n {
            return kw.err(format!("TYPE={ty} requires {n} constants, {} given", v.len()));
        }
        if v.len() > n + 1 {
            return kw.err(format!("TYPE={ty}: more than {n} constants (temperature-dependent data is not supported yet)"));
        }
        Ok(())
    };
    let mut lamina = None;
    let d = match ty.as_str() {
        "ENGINEERING CONSTANTS" => {
            need(9)?;
            let c: [f64; 9] = std::array::from_fn(|i| v[i]);
            engineering_constants(&c).or_else(|e| kw.err(e.0))?
        }
        "ORTHOTROPIC" => {
            need(9)?;
            let mut d = [[0.0; 6]; 6];
            d[0][0] = v[0];
            d[0][1] = v[1];
            d[1][1] = v[2];
            d[0][2] = v[3];
            d[1][2] = v[4];
            d[2][2] = v[5];
            d[3][3] = v[6];
            d[4][4] = v[7];
            d[5][5] = v[8];
            d[1][0] = d[0][1];
            d[2][0] = d[0][2];
            d[2][1] = d[1][2];
            d
        }
        "ANISOTROPIC" => {
            need(21)?;
            let mut d = [[0.0; 6]; 6];
            let mut k = 0;
            for j in 0..6 {
                for i in 0..=j {
                    d[i][j] = v[k];
                    d[j][i] = v[k];
                    k += 1;
                }
            }
            d
        }
        "LAMINA" => {
            need(6)?;
            let (e1, e2, n12, g12) = (v[0], v[1], v[2], v[3]);
            if !(e1 > 0.0 && e2 > 0.0 && g12 > 0.0) {
                return kw.err("E1, E2 and G12 must be positive");
            }
            let n21 = n12 * e2 / e1;
            let den = 1.0 - n12 * n21;
            if !(den > 0.0) {
                return kw.err("LAMINA: 1 − ν12 ν21 must be positive");
            }
            lamina = Some([e1 / den, n12 * e2 / den, 0.0, n12 * e2 / den, e2 / den, 0.0, 0.0, 0.0, g12]);
            let mut d = [[0.0; 6]; 6];
            d[3][3] = g12;
            d[4][4] = v[4];
            d[5][5] = v[5];
            d
        }
        other => return kw.err(format!("TYPE={other} is not supported (ISOTROPIC, ENGINEERING CONSTANTS, ORTHOTROPIC, ANISOTROPIC, LAMINA)")),
    };
    if lamina.is_none() {
        // stability: the matrix must be positive definite
        let flat: Vec<f64> = d.iter().flatten().copied().collect();
        if !positive_definite(&flat, 6) {
            return kw.err(format!("TYPE={ty}: the elasticity matrix is not positive definite"));
        }
    }
    Ok(Some(Arc::new(AnisotropicElastic { d, lamina, kind: ty })))
}

/// Cholesky test of positive definiteness.
pub fn positive_definite(a: &[f64], n: usize) -> bool {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) {
                    return false;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    true
}

impl MaterialModel for AnisotropicElastic {
    fn name(&self) -> &str {
        "ELASTIC (anisotropic)"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let d = self.dmat(input.mode)?;
        let stress = super::matvec(&d, input.strain);
        Ok(MatOutput { stress, tangent: d })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.dmat(mode)
    }
}

// ---------------------------------------------------------------------------
// *ORIENTATION
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationSystem {
    Rectangular,
    Cylindrical,
}

/// A local coordinate system for material directions.
#[derive(Debug, Clone)]
pub struct Orientation {
    pub name: String,
    pub system: OrientationSystem,
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub c: [f64; 3],
    /// Extra rotation (local axis 1–3, angle in degrees).
    pub extra: Option<(usize, f64)>,
}

impl Orientation {
    pub fn parse(kw: &RawKeyword) -> Result<Orientation> {
        let name = kw.param_upper("NAME").unwrap_or_default();
        let system = match kw.param_upper("SYSTEM").as_deref() {
            None | Some("RECTANGULAR") => OrientationSystem::Rectangular,
            Some("CYLINDRICAL") => OrientationSystem::Cylindrical,
            Some(o) => return kw.err(format!("SYSTEM={o} is not supported yet (RECTANGULAR, CYLINDRICAL)")),
        };
        match kw.param_upper("DEFINITION").as_deref() {
            None | Some("COORDINATES") => {}
            Some(o) => return kw.err(format!("DEFINITION={o} is not supported yet (only COORDINATES)")),
        }
        if kw.has("LOCAL DIRECTIONS") {
            return kw.err("LOCAL DIRECTIONS is not supported yet");
        }
        let recs = kw.records();
        let Some(first) = recs.first() else { return kw.err("data line with the points a and b is missing") };
        let v = kw.floats(first)?;
        if v.len() < 6 {
            return kw.err("data line 1 must have a (3 coordinates) and b (3 coordinates)");
        }
        let a = [v[0], v[1], v[2]];
        let b = [v[3], v[4], v[5]];
        let c = if v.len() >= 9 { [v[6], v[7], v[8]] } else { [0.0; 3] };
        let extra = match recs.get(1) {
            Some(d) => {
                let w = kw.floats(d)?;
                let axis = w.first().copied().unwrap_or(1.0).round() as usize;
                let ang = w.get(1).copied().unwrap_or(0.0);
                if !(1..=3).contains(&axis) {
                    return kw.err_at(d.line, "the axis of the extra rotation must be 1, 2 or 3");
                }
                // the axis is kept even with angle 0 (shells: it specifies which axes
                // lie in the shell plane, see elements/shell_section.rs)
                Some((axis, ang))
            }
            None => None,
        };
        let o = Orientation { name, system, a, b, c, extra };
        if o.system == OrientationSystem::Rectangular {
            let ax = o.axes_at([0.0; 3]);
            if ax.iter().any(|r| r.iter().any(|x| !x.is_finite())) {
                return kw.err("the points a, b and the origin do not give a coordinate system (parallel or coincident)");
            }
        }
        Ok(o)
    }

    /// Local axes at the point `p`: row i = local axis i expressed in global coordinates.
    pub fn axes_at(&self, p: [f64; 3]) -> [[f64; 3]; 3] {
        let (e1, e2, e3) = match self.system {
            OrientationSystem::Rectangular => {
                let x = sub(self.a, self.c);
                let e1 = scale(x, 1.0 / norm(x));
                let y = sub(self.b, self.c);
                let z = cross(e1, y);
                let e3 = scale(z, 1.0 / norm(z));
                (e1, cross(e3, e1), e3)
            }
            OrientationSystem::Cylindrical => {
                let ax = sub(self.b, self.a);
                let e3 = scale(ax, 1.0 / norm(ax));
                let d = sub(p, self.a);
                let r = sub(d, scale(e3, dot(d, e3)));
                let e1 = scale(r, 1.0 / norm(r));
                (e1, cross(e3, e1), e3)
            }
        };
        let mut e = [e1, e2, e3];
        if let Some((axis, ang)) = self.extra {
            let k = e[axis - 1];
            for (i, v) in e.iter_mut().enumerate() {
                if i != axis - 1 {
                    *v = rotate_vec(*v, k, ang);
                }
            }
        }
        e
    }

    /// Transformation matrix for strains in the stress state `mode`
    /// (ε_local = T ε_global, engineering shear strain), ncomp × ncomp row-major.
    pub fn strain_transform_mode(&self, p: [f64; 3], mode: StressMode) -> Vec<f64> {
        let t = strain_transform(&self.axes_at(p));
        let idx = mode_components(mode);
        idx.iter().flat_map(|&i| idx.iter().map(move |&j| t[i][j])).collect()
    }
}

/// Component pairs (i, j) for the 3D order 11, 22, 33, 12, 13, 23.
pub const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

/// The 6×6 matrix T with ε_local = T ε_global (engineering shear strain) for
/// the rotation `q` (row i = local axis i).
pub fn strain_transform(q: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut t = [[0.0; 6]; 6];
    for (jc, &(k, l)) in VOIGT.iter().enumerate() {
        // global strain tensor for the unit component jc
        let mut eg = [[0.0; 3]; 3];
        if k == l {
            eg[k][k] = 1.0;
        } else {
            eg[k][l] = 0.5;
            eg[l][k] = 0.5;
        }
        for (ir, &(i, j)) in VOIGT.iter().enumerate() {
            let mut v = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    v += q[i][a] * q[j][b] * eg[a][b];
                }
            }
            t[ir][jc] = if i == j { v } else { 2.0 * v };
        }
    }
    t
}

/// Orientation per section (the section's ORIENTATION= against *ORIENTATION in the model).
pub fn section_orientations(model: &FeModel) -> Result<Vec<Option<Orientation>>> {
    let mut out = Vec::with_capacity(model.sections.len());
    for s in &model.sections {
        let Some(name) = s.raw.param_upper("ORIENTATION") else {
            out.push(None);
            continue;
        };
        let kw = model.extra.iter().find(|k| k.name == "ORIENTATION" && k.param_upper("NAME").as_deref() == Some(name.as_str()));
        let Some(kw) = kw else {
            return s.raw.err(format!("the orientation '{name}' is not defined (*ORIENTATION, NAME={name})"));
        };
        out.push(Some(Orientation::parse(kw)?));
    }
    Ok(out)
}
