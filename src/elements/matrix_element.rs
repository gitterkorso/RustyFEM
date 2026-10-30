//! Matrix elements: linear elements with a given stiffness and mass matrix.
//!
//! Used for
//! - substructures (*ELEMENT, TYPE=Zn, FILE=job + *SUBSTRUCTURE PROPERTY):
//!   the condensed matrix from *SUBSTRUCTURE GENERATE
//!   (`steps/substructure.rs`). The element's nodes correspond to the retained nodes
//!   in ascending node number order. *SUBSTRUCTURE PROPERTY
//!   (translation on the first line, rotation a, b, angle on the second) rotates
//!   the matrices: K' = T K Tᵀ, where T rotates displacements and rotations
//!   node by node.
//! - matrices that are read in (*MATRIX INPUT + *MATRIX ASSEMBLE,
//!   `inp/keywords/infrastructure.rs`): one element over the matrix's nodes.
//!
//! The element types are created when the file is read and live for the rest of the process.

use super::api::{ElemCtx, ElementType, Family, MassKind};
use crate::fem::dof::Dof;
use crate::fem::model::RawKeyword;
use crate::inp::InpCtx;
use crate::model::{FemError, Result};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// No static types.
pub static TYPES: &[&dyn ElementType] = &[];

pub struct MatrixElement {
    pub name: &'static str,
    /// Degrees of freedom per node (ascending).
    pub dofs: Vec<&'static [Dof]>,
    /// Stiffness, dense and row-wise (n×n, n = number of degrees of freedom).
    pub k: Vec<f64>,
    pub m: Option<Vec<f64>>,
    pub dim: usize,
    /// Substructure: *SUBSTRUCTURE PROPERTY can rotate the matrices.
    pub substructure: bool,
}

impl MatrixElement {
    /// Creates (and leaks) an element type.
    pub fn leak(name: &str, dofs: Vec<Vec<Dof>>, k: Vec<f64>, m: Option<Vec<f64>>, substructure: bool) -> &'static MatrixElement {
        let dim = if dofs.iter().flatten().all(|d| matches!(d, 1 | 2 | 6 | 11)) { 2 } else { 3 };
        let dofs: Vec<&'static [Dof]> = dofs.into_iter().map(|d| &*Box::leak(d.into_boxed_slice())).collect();
        Box::leak(Box::new(MatrixElement { name: Box::leak(name.to_string().into_boxed_str()), dofs, k, m, dim, substructure }))
    }

    /// The rotation matrix (row-wise 3×3) from *SUBSTRUCTURE PROPERTY.
    fn rotation(&self, ctx: &ElemCtx) -> Result<Option<[[f64; 3]; 3]>> {
        if !self.substructure {
            return Ok(None);
        }
        let Some(sec) = ctx.section.filter(|s| s.raw.name == "SUBSTRUCTURE PROPERTY") else { return Ok(None) };
        let kw = &sec.raw;
        let Some(d) = kw.data.get(1) else { return Ok(None) };
        let v = kw.floats(d)?;
        if v.len() < 7 || v[6] == 0.0 {
            return Ok(None);
        }
        let axis = [v[3] - v[0], v[4] - v[1], v[5] - v[2]];
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len == 0.0 {
            return kw.err("the points a and b of the rotation axis coincide");
        }
        let mut r = [[0.0; 3]; 3];
        for j in 0..3 {
            let mut e = [0.0; 3];
            e[j] = 1.0;
            let c = crate::fem::model::rotate_vec(e, axis, v[6]);
            for i in 0..3 {
                r[i][j] = c[i];
            }
        }
        if self.dim == 2 && (axis[0].abs() > 1e-12 * len || axis[1].abs() > 1e-12 * len) {
            return kw.err("a planar substructure can only be rotated about the z axis");
        }
        Ok(Some(r))
    }

    /// T A Tᵀ with node-wise rotation of displacements (1–3) and rotations (4–6).
    fn rotate(&self, a: &[f64], r: &[[f64; 3]; 3]) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        let mut t = vec![0.0; n * n];
        let mut base = 0;
        for (node, d) in self.dofs.iter().enumerate() {
            let mut done = vec![false; d.len()];
            for group in [[1u8, 2, 3], [4, 5, 6]] {
                let idx: Vec<Option<usize>> = group.iter().map(|g| d.iter().position(|x| x == g)).collect();
                let present = idx.iter().filter(|i| i.is_some()).count();
                if present == 0 {
                    continue;
                }
                let planar = self.dim == 2;
                if planar && group[0] == 1 {
                    // plane rotation about z: U1, U2
                    let (Some(a1), Some(a2)) = (idx[0], idx[1]) else {
                        return Err(FemError(format!("{}: node {} lacks U1 or U2 and cannot be rotated", self.name, node + 1)));
                    };
                    for (p, ap) in [(0, a1), (1, a2)] {
                        for (q, aq) in [(0, a1), (1, a2)] {
                            t[(base + ap) * n + base + aq] = r[p][q];
                        }
                        done[ap] = true;
                    }
                    continue;
                }
                if planar {
                    continue;
                }
                if present != 3 {
                    return Err(FemError(format!("{}: node {} does not have all three degrees of freedom {:?} and cannot be rotated", self.name, node + 1, group)));
                }
                for p in 0..3 {
                    for q in 0..3 {
                        t[(base + idx[p].unwrap()) * n + base + idx[q].unwrap()] = r[p][q];
                    }
                    done[idx[p].unwrap()] = true;
                }
            }
            for (i, ok) in done.iter().enumerate() {
                if !ok {
                    t[(base + i) * n + base + i] = 1.0;
                }
            }
            base += d.len();
        }
        // T A Tᵀ
        let mut ta = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let tik = t[i * n + k];
                if tik != 0.0 {
                    for j in 0..n {
                        ta[i * n + j] += tik * a[k * n + j];
                    }
                }
            }
        }
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = (0..n).map(|k| ta[i * n + k] * t[j * n + k]).sum();
            }
        }
        Ok(out)
    }
}

impl ElementType for MatrixElement {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        self.dofs.len()
    }
    fn node_dofs(&self, i: usize) -> &'static [Dof] {
        self.dofs[i]
    }
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        match self.rotation(ctx)? {
            Some(r) => self.rotate(&self.k, &r),
            None => Ok(self.k.clone()),
        }
    }
    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        let m = self.m.clone().unwrap_or_else(|| vec![0.0; n * n]);
        match self.rotation(ctx)? {
            Some(r) => self.rotate(&m, &r),
            None => Ok(m),
        }
    }
    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(vec![0.0; self.n_nodes()])
    }
}

/// The element type for *ELEMENT, TYPE=Zn (substructure), or `None` if the type
/// is not Zn. The substructure is fetched from `steps::substructure::find`.
pub fn lookup_substructure(ctx: &InpCtx, kw: &RawKeyword, ty: &str) -> Result<Option<&'static dyn ElementType>> {
    let up = ty.trim().to_uppercase();
    if up.strip_prefix('Z').and_then(|s| s.parse::<u32>().ok()).is_none() {
        return Ok(None);
    }
    let file = kw.param("FILE").map(|s| s.trim().trim_matches('"').to_string());
    let lib = crate::steps::substructure::find(ctx, file.as_deref(), &up).map_err(|e| FemError(format!("{}: {}", kw.location(), e.0)))?;
    static CACHE: OnceLock<Mutex<HashMap<String, &'static MatrixElement>>> = OnceLock::new();
    let key = format!("{}|{}|{}", lib.source, up, lib.serial);
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = cache.get(&key) {
        return Ok(Some(*t));
    }
    let t = MatrixElement::leak(&up, lib.dofs.clone(), lib.k.clone(), lib.m.clone(), true);
    cache.insert(key, t);
    Ok(Some(t))
}
