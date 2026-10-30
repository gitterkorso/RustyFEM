//! Kinematics for geometrically nonlinear analysis (NLGEOM=YES).
//!
//! - `continuum`: updated Lagrangian with Hughes–Winget increments (CPS/CPE/C3D8
//!   and other continuum elements with integration points in the reference configuration).
//! - `truss`: trusses with logarithmic strain.
//! - `beam`: Euler–Bernoulli beams (B23, B33) with Hermite interpolation of the position.
//! - `hyperdual`: hyperdual numbers for exact derivatives of the strain energy.
//! - `pressure`: follower pressure on deformed surfaces.
//! - `tensor`: 3×3 tensors, polar decomposition, rotations.
//!
//! Elements call in here from their NLGEOM branch in `internal_force`. The
//! per-element state is the continuum core's usual layout (stress + material state
//! per integration point) followed by an extension read only by this module:
//! logarithmic strain (LE) and integrated strain per integration point, and last
//! the number of points and a marker. The stress is Cauchy stress in global axes,
//! so S can be output as usual.

pub mod bbar_tangent;
pub mod beam;
pub mod continuum;
pub mod fbar_hyper;
pub mod hyperdual;
pub mod pressure;
pub mod tensor;
pub mod truss;

use crate::elements::{Family, IpField};
use crate::fem::assembly::Prepared;
use crate::fem::output::FrameData;

/// Marker at the end of the element state for the NLGEOM extension.
pub const MAGIC: f64 = -7.357_913_e77;

/// Appends the extension (12 values per point: LE, integrated strain) to the state.
pub fn push_ext(state: &mut Vec<f64>, ext: &[f64], nips: usize) {
    state.extend_from_slice(ext);
    state.push(nips as f64);
    state.push(MAGIC);
}

/// The extension (LE for all points, then integrated strain), if present.
pub fn ext(state: &[f64]) -> Option<&[f64]> {
    let n = state.len();
    if n < 2 || state[n - 1] != MAGIC {
        return None;
    }
    let nips = state[n - 2] as usize;
    let len = 12 * nips;
    if n < len + 2 {
        return None;
    }
    Some(&state[n - 2 - len..n - 2])
}

/// Replaces the small-strain field E with logarithmic strain LE (output with NLGEOM)
/// for elements that have NLGEOM state.
pub fn fix_frame_output(p: &Prepared, frame: &mut FrameData, state: &[Vec<f64>]) {
    for (e, fields) in frame.elem.iter_mut().enumerate() {
        let Some(x) = state.get(e).and_then(|s| ext(s)) else { continue };
        let nips = x.len() / 12;
        let fam = p.model.elements[e].etype.family();
        let labels: &[&str] = match fam {
            Family::Truss => &["LE11"],
            Family::Continuum2D | Family::Axisymmetric => &["LE11", "LE22", "LE33", "LE12"],
            Family::Continuum3D => &["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"],
            _ => continue,
        };
        let mut le = IpField::new("LE", labels);
        for q in 0..nips {
            let v = &x[6 * q..6 * q + 6];
            // shear components as engineering strain
            let vals = match labels.len() {
                1 => vec![v[0]],
                4 => vec![v[0], v[1], v[2], 2.0 * v[3]],
                _ => vec![v[0], v[1], v[2], 2.0 * v[3], 2.0 * v[4], 2.0 * v[5]],
            };
            le.points.push((q + 1, 0, vals));
        }
        fields.retain(|f| f.name != "E" && f.name != "LE");
        fields.push(le);
    }
}
