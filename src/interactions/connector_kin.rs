//! Kopplingstypernas kinematik (inkluderas med `#[path]` från `connector.rs`).
//!
//! Varje kopplingselement har sex komponenter (1–3 förskjutning, 4–6
//! rotation). En komponent är tillgänglig (fri, kan ha
//! beteenden), låst (villkor) eller saknas. Kinematiken är linjäriserad (små
//! förskjutningar och rotationer, som implicit analys ger utan NLGEOM för små
//! laster) och varje komponent är en rad c med 12 koefficienter mot
//! (u_a, θ_a, u_b, θ_b) i globala axlar: komponentens värde (CU/CUR) är c·u.
//!
//! Beteckningar: e_i^a, e_i^b lokala axlar i nod a och b (*ORIENTATION, b får
//! a:s om den andra saknas), d = X_b − X_a, Δu = u_b − u_a, Δθ = θ_b − θ_a,
//! w = Δu + d × θ_a (b:s läge i a:s medroterande system).
//!
//! Translation (komponent 1–3):
//! - CARTESIAN: e_i^a · w, alla tillgängliga.
//! - PROJECTION CARTESIAN: e_i^a · (Δu + d × ½(θ_a + θ_b)).
//! - JOIN: e_i^a · w låsta (b följer a:s rotation). LINK: n · Δu låst (n = d/|d|).
//! - AXIAL: komponent 1 = n · Δu, läget |d|.
//! - SLOT: 1 tillgänglig, 2–3 låsta (e_i^a · w). SLIDE-PLANE: 1 låst, 2–3 tillgängliga.
//! - RADIAL-THRUST: 1 = e_r · w (radiell riktning i a:s 1–2-plan), 3 = e_3^a · w.
//!
//! Rotation (komponent 4–6):
//! - ALIGN: e_i^b · Δθ låsta. REVOLUTE: 4 = e_1^a · Δθ, 5–6 = e_2^b, e_3^b · Δθ låsta.
//! - UNIVERSAL: 4 = e_1^a · Δθ, 6 = e_3^b · Δθ, 5 = e_2^b · Δθ låst.
//! - CONSTANT VELOCITY: 5 = −ê · Δθ låst, ê = (e_3^a + e_3^b)/|…| (bisektrisen);
//!   4 och 6 skrivs ut som 0.
//! - ROTATION, PROJECTION FLEXION-TORSION: e_i^a · Δθ. FLEXION-TORSION:
//!   (e_2^a, e_3^a, e_1^a) · Δθ. Kräver samma axlar i a och b.
//! - CARDAN (X-Y-Z) och EULER (Z-X-Z): ändringen av vinklarna för den
//!   relativa rotationen R₀ = E_a E_bᵀ, dφ = J⁻¹ E_a Δθ. EULER med
//!   nutationsvinkeln β = 0 (samma axlar) är singulär; då skrivs 0.
//!
//! Sammansatta typer: BEAM (w och ALIGN låsta), WELD = JOIN + ALIGN,
//! HINGE = JOIN + REVOLUTE, UJOINT = JOIN + UNIVERSAL, CVJOINT = JOIN +
//! CONSTANT VELOCITY, CYLINDRICAL = SLOT + REVOLUTE, PLANAR = SLIDE-PLANE +
//! REVOLUTE, TRANSLATOR = SLOT + ALIGN, BUSHING = PROJECTION CARTESIAN +
//! PROJECTION FLEXION-TORSION.

/// Komponentens status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Saknas (skrivs ut som 0, inga beteenden).
    None,
    /// Tillgänglig: fri rörelse, kan ha beteenden.
    Available,
    /// Låst av kopplingstypen (villkor).
    Constrained,
}

/// En komponent: status, rad mot (u_a, θ_a, u_b, θ_b) och läget i referensläget.
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

/// 3×3-matrisen med kolumnerna c0, c1, c2 löst för höger led w.
fn solve3(c: [V3; 3], w: V3) -> Option<V3> {
    let det = dot(c[0], cross(c[1], c[2]));
    if det.abs() < 1e-10 {
        return None;
    }
    // Cramers regel
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

/// Rader för vinkelförändringar dφ = J⁻¹ E_a Δθ (J:s kolumner i a:s lokala
/// system) och vinklarna i referensläget. `None` om J är singulär.
fn angle_rows(ea: &[V3; 3], j: [V3; 3], angles: V3) -> Option<[CompDef; 3]> {
    let mut out = [CompDef::default(); 3];
    // kolumnvis: dφ_k = Σ_i (J⁻¹)_{k i} (e_i^a · Δθ)
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

/// Relativ rotation R₀ = E_a E_bᵀ i a:s lokala system (kolumn j = e_j^b).
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

/// Komponenterna för kopplingstyperna `types` (versaler). Fel som klartext.
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
            Err(format!("{name} med olika lokala axlar i nod a och b stöds inte ännu"))
        }
    };
    let mut set_trans = |c: &mut [CompDef; 6], name: &str| -> std::result::Result<(), String> {
        if trans_set {
            return Err(format!("två translationstyper ({name})"));
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
                    return Err("LINK kräver att noderna inte sammanfaller".into());
                }
                c[0] = comp(Constrained, trans(nd, d, 0.0, 0.0), n_link);
            }
            "AXIAL" => {
                if n_link == 0.0 {
                    return Err("AXIAL kräver att noderna inte sammanfaller".into());
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
                    return Err("RADIAL-THRUST kräver ett radiellt avstånd mellan noderna".into());
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
            return Err(format!("två rotationstyper ({name})"));
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
                    return Err("CONSTANT VELOCITY med motriktade axlar e3".into());
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
                let rows = angle_rows(ea, j, [p1, p2, p3]).ok_or("CARDAN är singulär (φ2 = ±90°) i referensläget")?;
                c[3..6].copy_from_slice(&rows);
            }
            "EULER" => {
                // R₀ = Rz(α) Rx(β) Rz(γ)
                let b = r0[2][2].clamp(-1.0, 1.0).acos();
                if b.sin().abs() < 1e-9 {
                    // singulär (samma axel e3): vinklarna skrivs som 0
                    for i in 0..3 {
                        c[3 + i] = comp(Available, [0.0; 12], 0.0);
                    }
                } else {
                    let a = r0[0][2].atan2(-r0[1][2]);
                    let g = r0[2][0].atan2(r0[2][1]);
                    let j = [[0.0, 0.0, 1.0], rz(a, [1.0, 0.0, 0.0]), rz(a, rx(b, [0.0, 0.0, 1.0]))];
                    let rows = angle_rows(ea, j, [a, b, g]).ok_or("EULER är singulär i referensläget")?;
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
                    "kopplingstypen {other} stöds inte ännu (AXIAL, CARTESIAN, JOIN, LINK, PROJECTION CARTESIAN, RADIAL-THRUST, SLIDE-PLANE, SLOT, ALIGN, CARDAN, CONSTANT VELOCITY, EULER, FLEXION-TORSION, PROJECTION FLEXION-TORSION, REVOLUTE, ROTATION, UNIVERSAL, BEAM, BUSHING, CVJOINT, CYLINDRICAL, HINGE, PLANAR, TRANSLATOR, UJOINT, WELD)"
                ))
            }
        }
    }
    Ok(c)
}
