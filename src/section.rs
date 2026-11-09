//! Beam cross-sections. The properties are exported unchanged to .inp
//! (*BEAM GENERAL SECTION), so the exported model gets exactly the same numbers.

/// Beam cross-section. `i11` is the moment of inertia for bending in the "height" direction
/// (local 2-axis, upwards), `i22` for lateral bending, `j` the torsional stiffness.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamSection {
    pub a: f64,
    pub i11: f64,
    pub i22: f64,
    pub j: f64,
    /// The shape for display: Abaqus section name and dimensions (e.g. ("RECT", [b, h])).
    /// Only used to draw the profile; the properties above are what the analysis uses.
    pub outline: Option<(String, Vec<f64>)>,
}

impl BeamSection {
    /// Rectangle with width `b` and height `h`.
    pub fn rect(b: f64, h: f64) -> Self {
        BeamSection { a: b * h, i11: b * h.powi(3) / 12.0, i22: h * b.powi(3) / 12.0, j: rect_torsion(b, h), outline: None }
    }

    /// Solid circle with radius `r`.
    pub fn circle(r: f64) -> Self {
        let i = std::f64::consts::PI * r.powi(4) / 4.0;
        BeamSection { a: std::f64::consts::PI * r * r, i11: i, i22: i, j: 2.0 * i, outline: None }
    }

    /// Pipe with outer radius `r` and wall thickness `t`.
    pub fn pipe(r: f64, t: f64) -> Self {
        let ri = r - t;
        let pi = std::f64::consts::PI;
        let i = pi * (r.powi(4) - ri.powi(4)) / 4.0;
        BeamSection { a: pi * (r * r - ri * ri), i11: i, i22: i, j: 2.0 * i, outline: None }
    }

    /// Thin-walled pipe with outer radius `r` and wall thickness `t` (like
    /// *BEAM SECTION, SECTION=PIPE): I = π r_m³ t and J = 2I with the mean radius r_m.
    pub fn pipe_thin(r: f64, t: f64) -> Self {
        let rm = r - t / 2.0;
        let pi = std::f64::consts::PI;
        let i = pi * rm.powi(3) * t;
        BeamSection { a: 2.0 * pi * rm * t, i11: i, i22: i, j: 2.0 * i, outline: None }
    }

    /// I-profile: height `h`, flange width `b`, flange thickness `tf`, web thickness `tw`.
    pub fn ibeam(h: f64, b: f64, tf: f64, tw: f64) -> Self {
        let hw = h - 2.0 * tf;
        BeamSection {
            a: 2.0 * b * tf + hw * tw,
            i11: (b * h.powi(3) - (b - tw) * hw.powi(3)) / 12.0,
            i22: 2.0 * tf * b.powi(3) / 12.0 + hw * tw.powi(3) / 12.0,
            j: (2.0 * b * tf.powi(3) + hw * tw.powi(3)) / 3.0,
            outline: None,
        }
    }

    /// The same section with a display shape (see `outline`).
    pub fn with_outline(self, shape: &str, dims: &[f64]) -> Self {
        BeamSection { outline: Some((shape.to_string(), dims.to_vec())), ..self }
    }

    /// Arbitrary cross-section from tabulated values.
    pub fn general(a: f64, i11: f64, i22: f64, j: f64) -> Self {
        BeamSection { a, i11, i22, j, outline: None }
    }
}

/// Torsional stiffness of a rectangle (Roark, approximate).
fn rect_torsion(b: f64, h: f64) -> f64 {
    let (a, c) = if b >= h { (b, h) } else { (h, b) };
    a * c.powi(3) * (1.0 / 3.0 - 0.21 * (c / a) * (1.0 - c.powi(4) / (12.0 * a.powi(4))))
}
