//! Kinematics of the connection types (included with `#[path]` from `connector.rs`).
//!
//! Each connector element has six components (1–3 displacement, 4–6
//! rotation). A component is available (free, may have behaviors),
//! constrained (locked) or absent. The kinematics are linearized (small
//! displacements and rotations, as implicit analysis without NLGEOM gives for
//! small loads) and each component is a row c with 12 coefficients against
//! (u_a, θ_a, u_b, θ_b) in global axes: the component value (CU/CUR) is c·u.
//!
//! Notation: e_i^a, e_i^b local axes at node a and b (*ORIENTATION, b gets
//! a's if the other one is missing), d = X_b − X_a, Δu = u_b − u_a, Δθ = θ_b − θ_a,
//! w = Δu + d × θ_a (b's position in a's corotating system).
//!
//! Translation (components 1–3):
//! - CARTESIAN: e_i^a · w, all available.
//! - PROJECTION CARTESIAN: e_i^a · (Δu + d × ½(θ_a + θ_b)).
//! - JOIN: e_i^a · w constrained (b follows a's rotation). LINK: n · Δu constrained (n = d/|d|).
//! - AXIAL: component 1 = n · Δu, position |d|.
//! - SLOT: 1 available, 2–3 constrained (e_i^a · w). SLIDE-PLANE: 1 constrained, 2–3 available.
//! - RADIAL-THRUST: 1 = e_r · w (radial direction in a's 1–2 plane), 3 = e_3^a · w.
//!
//! Rotation (components 4–6):
//! - ALIGN: e_i^b · Δθ constrained. REVOLUTE: 4 = e_1^a · Δθ, 5–6 = e_2^b, e_3^b · Δθ constrained.
//! - UNIVERSAL: 4 = e_1^a · Δθ, 6 = e_3^b · Δθ, 5 = e_2^b · Δθ constrained.
//! - CONSTANT VELOCITY: 5 = −ê · Δθ constrained, ê = (e_3^a + e_3^b)/|…| (the bisector);
//!   4 and 6 are output as 0.
//! - ROTATION, PROJECTION FLEXION-TORSION: e_i^a · Δθ. FLEXION-TORSION:
//!   (e_2^a, e_3^a, e_1^a) · Δθ. Requires the same axes at a and b.
//! - CARDAN (X-Y-Z) and EULER (Z-X-Z): the change of the angles of the
//!   relative rotation R₀ = E_a E_bᵀ, dφ = J⁻¹ E_a Δθ. EULER with the
//!   nutation angle β = 0 (same axes) is singular; then 0 is written.
//!
//! Assembled types: BEAM (w and ALIGN constrained), WELD = JOIN + ALIGN,
//! HINGE = JOIN + REVOLUTE, UJOINT = JOIN + UNIVERSAL, CVJOINT = JOIN +
//! CONSTANT VELOCITY, CYLINDRICAL = SLOT + REVOLUTE, PLANAR = SLIDE-PLANE +
//! REVOLUTE, TRANSLATOR = SLOT + ALIGN, BUSHING = PROJECTION CARTESIAN +
//! PROJECTION FLEXION-TORSION.

/// Component status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Absent (output as 0, no behaviors).
    None,
    /// Available: free motion, may have behaviors.
    Available,
    /// Constrained by the connection type (locked).
    Constrained,
}

/// A component: status, row against (u_a, θ_a, u_b, θ_b) and the position in the reference configuration.
#[derive(Debug, Clone, Copy)]
pub struct CompDef {
    pub status: Status,
    pub row: [f64; 12],
    pub p0: f64,
}

impl Default for CompDef {
    fn default() -> Self {
        CompDef { status: Status::None, row: [0.0; 12], p0: 0.0 }
    }
}

type V3 = [f64; 3];

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// e · Δu (+ θ_a·(e × d)·fa + θ_b·(e × d)·fb).
fn trans(e: V3, d: V3, fa: f64, fb: f64) -> [f64; 12] {
    let mut r = [0.0; 12];
    let ed = cross(e, d);
    for q in 0..3 {
        r[q] = -e[q];
        r[6 + q] = e[q];
        r[3 + q] = fa * ed[q];
        r[9 + q] = fb * ed[q];
    }
    r
}

/// e · Δθ.
fn rot(e: V3) -> [f64; 12] {
    let mut r = [0.0; 12];
    for q in 0..3 {
        r[3 + q] = -e[q];
        r[9 + q] = e[q];
    }
    r
}

fn comp(status: Status, row: [f64; 12], p0: f64) -> CompDef {
    CompDef { status, row, p0 }
}

/// The 3×3 matrix with columns c0, c1, c2 solved for the right-hand side w.
fn solve3(c: [V3; 3], w: V3) -> Option<V3> {
    let det = dot(c[0], cross(c[1], c[2]));
    if det.abs() < 1e-10 {
        return None;
    }
    // Cramer's rule
    let x0 = dot(w, cross(c[1], c[2])) / det;
    let x1 = dot(c[0], cross(w, c[2])) / det;
    let x2 = dot(c[0], cross(c[1], w)) / det;
    Some([x0, x1, x2])
}

fn rx(t: f64, v: V3) -> V3 {
    let (c, s) = (t.cos(), t.sin());
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}
fn ry(t: f64, v: V3) -> V3 {
    let (c, s) = (t.cos(), t.sin());
    [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]
}
fn rz(t: f64, v: V3) -> V3 {
    let (c, s) = (t.cos(), t.sin());
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

/// Rows for the angle changes dφ = J⁻¹ E_a Δθ (J's columns in a's local
/// system) and the angles in the reference configuration. `None` if J is singular.
fn angle_rows(ea: &[V3; 3], j: [V3; 3], angles: V3) -> Option<[CompDef; 3]> {
    let mut out = [CompDef::default(); 3];
    // column-wise: dφ_k = Σ_i (J⁻¹)_{k i} (e_i^a · Δθ)
    let mut inv = [[0.0; 3]; 3];
    for i in 0..3 {
        let mut w = [0.0; 3];
        w[i] = 1.0;
        let x = solve3(j, w)?;
        for k in 0..3 {
            inv[k][i] = x[k];
        }
    }
    for k in 0..3 {
        let mut e = [0.0; 3];
        for i in 0..3 {
            for q in 0..3 {
                e[q] += inv[k][i] * ea[i][q];
            }
        }
        out[k] = comp(Status::Available, rot(e), angles[k]);
    }
    Some(out)
}

/// Relative rotation R₀ = E_a E_bᵀ in a's local system (column j = e_j^b).
fn relative(ea: &[V3; 3], eb: &[V3; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = dot(ea[i], eb[j]);
        }
    }
    r
}

fn same_axes(ea: &[V3; 3], eb: &[V3; 3]) -> bool {
    (0..3).all(|i| (dot(ea[i], eb[i]) - 1.0).abs() < 1e-9)
}

/// The components for the connection types `types` (upper case). Errors as plain text.
pub fn components(types: &[String], d: V3, ea: &[V3; 3], eb: &[V3; 3]) -> std::result::Result<[CompDef; 6], String> {
    use Status::*;
    let mut c = [CompDef::default(); 6];
    let mut trans_set = false;
    let mut rot_set = false;
    let n_link = norm(d);
    let nd = if n_link > 0.0 { [d[0] / n_link, d[1] / n_link, d[2] / n_link] } else { [0.0; 3] };
    let dl = [dot(ea[0], d), dot(ea[1], d), dot(ea[2], d)];
    let need_same = |name: &str| -> std::result::Result<(), String> {
        if same_axes(ea, eb) {
            Ok(())
        } else {
            Err(format!("{name} with different local axes at node a and b is not supported yet"))
        }
    };
    let mut set_trans = |c: &mut [CompDef; 6], name: &str| -> std::result::Result<(), String> {
        if trans_set {
            return Err(format!("two translation types ({name})"));
        }
        trans_set = true;
        match name {
            "CARTESIAN" => {
                for i in 0..3 {
                    c[i] = comp(Available, trans(ea[i], d, 1.0, 0.0), dl[i]);
                }
            }
            "PROJECTION CARTESIAN" => {
                need_same(name)?;
                for i in 0..3 {
                    c[i] = comp(Available, trans(ea[i], d, 0.5, 0.5), dl[i]);
                }
            }
            "JOIN" => {
                for i in 0..3 {
                    c[i] = comp(Constrained, trans(ea[i], d, 1.0, 0.0), dl[i]);
                }
            }
            "LINK" => {
                if n_link == 0.0 {
                    return Err("LINK requires the nodes not to coincide".into());
                }
                c[0] = comp(Constrained, trans(nd, d, 0.0, 0.0), n_link);
            }
            "AXIAL" => {
                if n_link == 0.0 {
                    return Err("AXIAL requires the nodes not to coincide".into());
                }
                c[0] = comp(Available, trans(nd, d, 0.0, 0.0), n_link);
            }
            "SLOT" => {
                c[0] = comp(Available, trans(ea[0], d, 1.0, 0.0), dl[0]);
                for i in 1..3 {
                    c[i] = comp(Constrained, trans(ea[i], d, 1.0, 0.0), dl[i]);
                }
            }
            "SLIDE-PLANE" => {
                c[0] = comp(Constrained, trans(ea[0], d, 1.0, 0.0), dl[0]);
                for i in 1..3 {
                    c[i] = comp(Available, trans(ea[i], d, 1.0, 0.0), dl[i]);
                }
            }
            "RADIAL-THRUST" => {
                let r = (dl[0] * dl[0] + dl[1] * dl[1]).sqrt();
                if r == 0.0 {
                    return Err("RADIAL-THRUST requires a radial distance between the nodes".into());
                }
                let er = [0, 1, 2].map(|q| (dl[0] * ea[0][q] + dl[1] * ea[1][q]) / r);
                c[0] = comp(Available, trans(er, d, 1.0, 0.0), r);
                c[2] = comp(Available, trans(ea[2], d, 1.0, 0.0), dl[2]);
            }
            _ => unreachable!(),
        }
        Ok(())
    };
    let mut set_rot = |c: &mut [CompDef; 6], name: &str| -> std::result::Result<(), String> {
        if rot_set {
            return Err(format!("two rotation types ({name})"));
        }
        rot_set = true;
        let r0 = relative(ea, eb);
        match name {
            "ALIGN" => {
                for i in 0..3 {
                    c[3 + i] = comp(Constrained, rot(eb[i]), 0.0);
                }
            }
            "REVOLUTE" => {
                c[3] = comp(Available, rot(ea[0]), 0.0);
                c[4] = comp(Constrained, rot(eb[1]), 0.0);
                c[5] = comp(Constrained, rot(eb[2]), 0.0);
            }
            "UNIVERSAL" => {
                c[3] = comp(Available, rot(ea[0]), 0.0);
                c[4] = comp(Constrained, rot(eb[1]), 0.0);
                c[5] = comp(Available, rot(eb[2]), 0.0);
            }
            "CONSTANT VELOCITY" => {
                let s = [0, 1, 2].map(|q| ea[2][q] + eb[2][q]);
                let n = norm(s);
                if n < 1e-9 {
                    return Err("CONSTANT VELOCITY with opposite e3 axes".into());
                }
                c[4] = comp(Constrained, rot([-s[0] / n, -s[1] / n, -s[2] / n]), 0.0);
            }
            "ROTATION" | "PROJECTION FLEXION-TORSION" => {
                need_same(name)?;
                for i in 0..3 {
                    c[3 + i] = comp(Available, rot(ea[i]), 0.0);
                }
            }
            "FLEXION-TORSION" => {
                need_same(name)?;
                for (k, i) in [1, 2, 0].into_iter().enumerate() {
                    c[3 + k] = comp(Available, rot(ea[i]), 0.0);
                }
            }
            "CARDAN" => {
                // R₀ = Rx(φ1) Ry(φ2) Rz(φ3)
                let p2 = r0[0][2].clamp(-1.0, 1.0).asin();
                let p1 = (-r0[1][2]).atan2(r0[2][2]);
                let p3 = (-r0[0][1]).atan2(r0[0][0]);
                let j = [[1.0, 0.0, 0.0], rx(p1, [0.0, 1.0, 0.0]), rx(p1, ry(p2, [0.0, 0.0, 1.0]))];
                let rows = angle_rows(ea, j, [p1, p2, p3]).ok_or("CARDAN is singular (φ2 = ±90°) in the reference configuration")?;
                c[3..6].copy_from_slice(&rows);
            }
            "EULER" => {
                // R₀ = Rz(α) Rx(β) Rz(γ)
                let b = r0[2][2].clamp(-1.0, 1.0).acos();
                if b.sin().abs() < 1e-9 {
                    // singular (same e3 axis): the angles are written as 0
                    for i in 0..3 {
                        c[3 + i] = comp(Available, [0.0; 12], 0.0);
                    }
                } else {
                    let a = r0[0][2].atan2(-r0[1][2]);
                    let g = r0[2][0].atan2(r0[2][1]);
                    let j = [[0.0, 0.0, 1.0], rz(a, [1.0, 0.0, 0.0]), rz(a, rx(b, [0.0, 0.0, 1.0]))];
                    let rows = angle_rows(ea, j, [a, b, g]).ok_or("EULER is singular in the reference configuration")?;
                    c[3..6].copy_from_slice(&rows);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    };
    for t in types {
        match t.as_str() {
            "CARTESIAN" | "PROJECTION CARTESIAN" | "JOIN" | "LINK" | "AXIAL" | "SLOT" | "SLIDE-PLANE" | "RADIAL-THRUST" => set_trans(&mut c, t)?,
            "ALIGN" | "REVOLUTE" | "UNIVERSAL" | "CONSTANT VELOCITY" | "ROTATION" | "PROJECTION FLEXION-TORSION" | "FLEXION-TORSION" | "CARDAN" | "EULER" => set_rot(&mut c, t)?,
            "BEAM" => {
                set_trans(&mut c, "JOIN")?;
                set_rot(&mut c, "ALIGN")?;
            }
            "WELD" => {
                set_trans(&mut c, "JOIN")?;
                set_rot(&mut c, "ALIGN")?;
            }
            "HINGE" => {
                set_trans(&mut c, "JOIN")?;
                set_rot(&mut c, "REVOLUTE")?;
            }
            "UJOINT" => {
                set_trans(&mut c, "JOIN")?;
                set_rot(&mut c, "UNIVERSAL")?;
            }
            "CVJOINT" => {
                set_trans(&mut c, "JOIN")?;
                set_rot(&mut c, "CONSTANT VELOCITY")?;
            }
            "CYLINDRICAL" => {
                set_trans(&mut c, "SLOT")?;
                set_rot(&mut c, "REVOLUTE")?;
            }
            "PLANAR" => {
                set_trans(&mut c, "SLIDE-PLANE")?;
                set_rot(&mut c, "REVOLUTE")?;
            }
            "TRANSLATOR" => {
                set_trans(&mut c, "SLOT")?;
                set_rot(&mut c, "ALIGN")?;
            }
            "BUSHING" => {
                set_trans(&mut c, "PROJECTION CARTESIAN")?;
                set_rot(&mut c, "PROJECTION FLEXION-TORSION")?;
            }
            other => {
                return Err(format!(
                    "connection type {other} is not supported yet (AXIAL, CARTESIAN, JOIN, LINK, PROJECTION CARTESIAN, RADIAL-THRUST, SLIDE-PLANE, SLOT, ALIGN, CARDAN, CONSTANT VELOCITY, EULER, FLEXION-TORSION, PROJECTION FLEXION-TORSION, REVOLUTE, ROTATION, UNIVERSAL, BEAM, BUSHING, CVJOINT, CYLINDRICAL, HINGE, PLANAR, TRANSLATOR, UJOINT, WELD)"
                ))
            }
        }
    }
    Ok(c)
}
