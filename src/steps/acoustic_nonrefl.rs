//! Område gaps1: impedans- och strålningsränder för de akustiska stegen
//! (inkluderas från `acoustic.rs`).
//!
//! Strålningsvillkoret är n⁻·∇p = f (i k̃ + β) p med
//!
//! | NONREFLECTING | f | β |
//! |---|---|---|
//! | PLANAR | 1 | 0 |
//! | CIRCULAR | 1 | 1/(2r₁) |
//! | SPHERICAL | 1 | 1/r₁ |
//! | ELLIPTICAL | √((1 − ε²)/(1 − ε² ((x − x₀)·e_m/a)²)) | 1/(2a) |
//! | PROLATE SPHEROIDAL | som ELLIPTICAL | 1/a |
//!
//! Geometrifaktorn f varierar över randen och integreras in i randmatrisen,
//! H = ∫ f N N dS (x i ytans integrationspunkter). Dataraden för ELLIPTICAL
//! och PROLATE SPHEROIDAL är (yta, a, ε, x₀, y₀, z₀, e_m,x, e_m,y, e_m,z);
//! riktningen normeras. NONREFLECTING=IMPROVED (Sandler: planvåg med
//! godtycklig infallsvinkel, faktorn cos θ) är bara tillåten i transient
//! dynamik och byggs som PLANAR med en markering; `acoustic_improved.rs`
//! räknar cos θ i varje inkrement.

use super::*;

/// Geometrifaktorn för en strålningsrand.
#[derive(Debug, Clone, Copy)]
pub(super) enum GeoFactor {
    One,
    /// Ellips/sfäroid: halva storaxeln a, excentricitet ε, centrum, storaxelns riktning.
    Ellipse { a: f64, e: f64, x0: [f64; 3], em: [f64; 3] },
}

impl GeoFactor {
    pub(super) fn at(&self, x: [f64; 3]) -> f64 {
        match *self {
            GeoFactor::One => 1.0,
            GeoFactor::Ellipse { a, e, x0, em } => {
                let s = ((x[0] - x0[0]) * em[0] + (x[1] - x0[1]) * em[1] + (x[2] - x0[2]) * em[2]) / a;
                let d = 1.0 - e * e * s * s;
                if d > 0.0 { ((1.0 - e * e) / d).sqrt() } else { 1.0 }
            }
        }
    }
}

/// Randens slag och geometrifaktor ur nyckelordet och dataradens tal.
fn kind_of(m: &FeModel, kw: &RawKeyword, extra: &[f64]) -> Result<(ImpKind, GeoFactor, bool)> {
    if let Some(name) = kw.param("PROPERTY") {
        return Ok((property_table(m, name.trim(), kw)?, GeoFactor::One, false));
    }
    let nr = kw.param_upper("NONREFLECTING").unwrap_or_else(|| "PLANAR".into());
    let first = extra.first().copied().filter(|r| *r > 0.0);
    match nr.as_str() {
        "PLANAR" => Ok((ImpKind::Radiation(0.0), GeoFactor::One, false)),
        "IMPROVED" => Ok((ImpKind::Radiation(0.0), GeoFactor::One, true)),
        "CIRCULAR" | "SPHERICAL" => {
            let Some(r) = first else { return kw.err(format!("NONREFLECTING={nr} kräver radien r₁ på dataraden")) };
            Ok((ImpKind::Radiation(if nr == "CIRCULAR" { 0.5 / r } else { 1.0 / r }), GeoFactor::One, false))
        }
        "ELLIPTICAL" | "PROLATE SPHEROIDAL" => {
            let Some(a) = first else { return kw.err(format!("NONREFLECTING={nr} kräver halva storaxeln a på dataraden")) };
            let e = extra.get(1).copied().unwrap_or(0.0);
            if !(0.0..1.0).contains(&e) {
                return kw.err(format!("NONREFLECTING={nr}: excentriciteten ska ligga i [0, 1)"));
            }
            let g = |i: usize| extra.get(i).copied().unwrap_or(0.0);
            let x0 = [g(2), g(3), g(4)];
            let mut em = [g(5), g(6), g(7)];
            let n = (em[0] * em[0] + em[1] * em[1] + em[2] * em[2]).sqrt();
            if n > 0.0 {
                em.iter_mut().for_each(|v| *v /= n);
            } else {
                em = [1.0, 0.0, 0.0];
            }
            let beta = if nr == "ELLIPTICAL" { 0.5 / a } else { 1.0 / a };
            // geometrifaktorn används inte: lösningen beror inte på
            // ε, centrum eller storaxelns riktning och är densamma som för
            // CIRCULAR/SPHERICAL med r₁ = a (kontrollerat i *STEADY STATE
            // DYNAMICS och *DYNAMIC). Faktorn finns kvar för RUSTYFEM_ELLIPSE_F=1.
            let geo = if std::env::var("RUSTYFEM_ELLIPSE_F").is_ok_and(|v| v == "1") { GeoFactor::Ellipse { a, e, x0, em } } else { GeoFactor::One };
            Ok((ImpKind::Radiation(beta), geo, false))
        }
        other => kw.err(format!("NONREFLECTING={other} stöds inte (PLANAR, IMPROVED, CIRCULAR, SPHERICAL, ELLIPTICAL, PROLATE SPHEROIDAL)")),
    }
}

/// Talen efter ytan/elementet på en datarad.
fn numbers(kw: &RawKeyword, f: &[&str]) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    for s in f {
        if s.is_empty() {
            v.push(0.0);
            continue;
        }
        v.push(parse_f64(s).ok_or_else(|| FemError(format!("{}: '{s}' är inget tal", kw.location())))?);
    }
    while v.last() == Some(&0.0) && v.len() > 1 {
        v.pop();
    }
    Ok(v)
}

impl<'a, 'm> Sys<'a, 'm> {
    /// Bygger impedans- och strålningsränderna (`ImpBoundary`) för steg `index`.
    pub(super) fn build_impedances_gaps1(&mut self, index: usize) -> Result<()> {
        let m = self.p.model;
        let na = self.n_all();
        let transient = m.steps.get(index).is_some_and(|s| s.procedure_name() == "DYNAMIC");
        for kw in impedance_keywords(m, index) {
            // ytor: (element, ytindex, dataradens tal)
            let mut faces: Vec<(usize, usize, Vec<f64>)> = Vec::new();
            for d in kw.records() {
                let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).collect();
                if f.is_empty() || f[0].is_empty() {
                    continue;
                }
                if kw.name == "SIMPEDANCE" {
                    let extra = numbers(kw, &f[1..])?;
                    let surf = geom::surface(m, kw, f[0])?;
                    if surf.kind == "NODE" {
                        return kw.err("*SIMPEDANCE kräver en elementbaserad yta");
                    }
                    for (e, lab) in &surf.faces {
                        let i = lab.strip_prefix('S').and_then(|x| x.parse::<usize>().ok()).filter(|&i| i >= 1);
                        let Some(i) = i else { return kw.err(format!("ytan {lab} stöds inte för impedans")) };
                        faces.push((*e, i - 1, extra.clone()));
                    }
                } else {
                    let Some(lab) = f.get(1) else { return kw.err_at(d.line, "skriv 'element eller elementset, In'") };
                    let up = lab.to_uppercase();
                    let i = up.strip_prefix('I').and_then(|x| x.parse::<usize>().ok()).filter(|&i| i >= 1);
                    let Some(i) = i else { return kw.err_at(d.line, format!("'{lab}' är ingen impedansyta (I1, I2, ...)")) };
                    let extra = numbers(kw, &f[2..])?;
                    for e in geom::elem_ref(m, kw, f[0])? {
                        faces.push((e, i - 1, extra.clone()));
                    }
                }
            }
            // en rand per (nyckelord, datarad, material)
            let mut groups: Vec<(Vec<f64>, AcousticProps, Gen, ImpKind, bool)> = Vec::new();
            let mut improved_faces: Vec<(usize, usize)> = Vec::new();
            for (e, fi, extra) in faces {
                let el = &m.elements[e];
                let Some(a) = ac::lookup(el.etype.name()) else {
                    return kw.err(format!("{} är inget akustiskt element", m.elem_name(e)));
                };
                let props = material_props(m, e)?;
                let (kind, geo, improved) = kind_of(m, kw, &extra)?;
                if improved && !transient {
                    return kw.err("NONREFLECTING=IMPROVED kan bara användas i transient dynamik (*DYNAMIC)");
                }
                let x: Vec<[f64; 3]> = el.nodes.iter().map(|&nd| m.nodes[nd]).collect();
                let he = face_matrix_w(a, &x, fi, thickness(m, e), &|xp| geo.at(xp))?;
                if improved {
                    improved_faces.push((e, fi));
                }
                let gi = match groups.iter().position(|g| g.0 == extra && g.1.name == props.name) {
                    Some(i) => i,
                    None => {
                        groups.push((extra.clone(), props, Gen::new(na), kind, improved));
                        groups.len() - 1
                    }
                };
                let mut g = std::mem::take(&mut groups[gi].2);
                self.scatter(e, &he, &mut g);
                groups[gi].2 = g;
            }
            for (_, props, mut h, kind, improved) in groups {
                if improved {
                    continue;
                }
                h.finish();
                self.imp.push(ImpBoundary { kind, h, props });
            }
            if !improved_faces.is_empty() {
                // ränder med cos θ (acoustic_improved.rs): H byggs om i varje inkrement
                self.improved.push(super::improved::Improved { faces: improved_faces });
            }
        }
        Ok(())
    }
}
