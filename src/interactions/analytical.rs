//! Kontakt mot analytiska stela ytor: *SURFACE, TYPE=SEGMENTS (plana och
//! axisymmetriska modeller), TYPE=CYLINDER och TYPE=REVOLUTION (3D) med
//! *RIGID BODY, ANALYTICAL SURFACE=, REF NODE= som masteryta i *CONTACT PAIR.
//!
//! Geometri:
//! - profilen är sammanhängande segment i ett plan: START, LINE (slutpunkt),
//!   CIRCL (slutpunkt, centrum; den kortare bågen, < 179,74°) och PARAB
//!   (mittpunkt, slutpunkt; andragradskurva med mittpunkten i parametern ½),
//! - SEGMENTS: globala (X, Y) eller (r, z); CYLINDER: lokala (x, y) med origo a,
//!   x-axeln mot b och genereringsvektorn a → c längs negativa lokala z
//!   (c justeras vinkelrätt mot ab), oändlig längs genereringsvektorn;
//!   REVOLUTION: (r, z) med origo a och symmetriaxeln a → b (lokala z),
//! - utåtnormalen är n = e₃ × s (s = profilens riktning, e₃ = planets normal),
//!   dvs. riktningen vriden 90° moturs i profilplanet,
//! - inne i en *PART: profilen i partens system (CYLINDER: a = origo, b på
//!   x-axeln, c på negativa z-axeln; REVOLUTION: a = origo, r längs x, axeln
//!   längs y), placerad med instansens förflyttning och rotation.
//!
//! Ytan följer referensnoden R exakt (även utan NLGEOM, som brukligt för
//! analytiska ytor): x = x_R + R(θ_R)(X − X_R). Slavnoden projiceras på ytan i
//! kroppens referensläge, p_b = X_R + Rᵀ(x_s − x_R); närmaste punkten q och
//! normalen n förs tillbaka med R. Gapet är g = n · (x_s − q).
//!
//! Diskretisering: nod mot yta (standard för *CONTACT PAIR i implicit analys).
//! Finite sliding: närmaste punkt i varje utvärdering; SMALL SLIDING:
//! tangentplanet i slavnodens projektion i referensläget, som följer kroppen.
//! Normalkontakt och friktion som i `contact.rs` (HARD med styv
//! straffkoefficient 1e8 · elementstyvhet/area mot stel yta, PENALTY,
//! LINEAR/EXPONENTIAL, Coulomb med straffreglerad fastläge och kritisk elastisk
//! glidning SLIP TOLERANCE · slavytornas medelstorlek). Den relativa glidningen
//! räknas i kroppens system (slavnodens rörelse relativt ytan).
//!
//! Variationen δg = n · (δu_s − δu_R − δw × a) med a = q − x_R ger krafterna
//! f_s = A(−p n + τ), f_R = −f_s och M_R = −a × f_s på referensnoden och
//! tangenten A Gᵀ (p' n nᵀ + K_t) G (utan ytans krökningsterm, som bara
//! påverkar konvergensen).
//!
//! Begränsningar: inte SURFACE TO SURFACE, ingen FILLET RADIUS, inga
//! användarytor (TYPE=USER/RSURFU), ingen *CLEARANCE.

use super::contact::{self, PairProps};
use super::geom::{self, Facet};
use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::elements::common::{cross, dot, norm, scale, sub};
use crate::fem::dof::{Dof, NodeVec, U1, UR1, UR3};
use crate::fem::model::{parse_f64, FeModel, Placement, RawKeyword};
use crate::kinematics::tensor;
use crate::model::{fail, Result};
use std::sync::Mutex;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CONTACT PAIR"], build }];

/// Straffkoefficient för hård kontakt (direkt metod) mot en analytisk yta,
/// relativt slavsidans elementstyvhet. Lägre än för stela element (1e8): med
/// krökta ytor och koordinater av storleken 1 ger avrundningen i närmaste
/// punkten (~1e-16) annars ett kraftbrus över konvergenskraven, och gapets
/// linjäriseringsfel vid glidning blir för stort. Genomträngningen blir
/// ~1e-7 av deformationen.
const HARD_FACTOR_ANALYTICAL: f64 = 1e7;

// ---------------------------------------------------------------- namn och placering

/// Placeringen som parameter (`_PLACE`) för nyckelord från en part.
pub fn encode_placement(p: &Placement) -> String {
    let t = p.translation;
    match p.rotation {
        Some((a, b, d)) => format!("{};{};{};{};{};{};{};{};{};{}", t[0], t[1], t[2], a[0], a[1], a[2], b[0], b[1], b[2], d),
        None => format!("{};{};{}", t[0], t[1], t[2]),
    }
}

fn decode_placement(s: &str) -> Placement {
    let v: Vec<f64> = s.split(';').filter_map(|x| x.trim().parse::<f64>().ok()).collect();
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let rotation = (v.len() >= 10).then(|| ([g(3), g(4), g(5)], [g(6), g(7), g(8)], g(9)));
    Placement { translation: [g(0), g(1), g(2)], rotation }
}

/// Instansens placering för ett nyckelord från en part (ingen om det står på modellnivå).
pub fn placement_of(kw: &RawKeyword) -> Option<Placement> {
    kw.param("_PLACE").map(decode_placement)
}

/// Ett namn i nyckelordet: med instansprefix om nyckelordet kommer från en part.
pub fn scoped(kw: &RawKeyword, token: &str) -> String {
    let t = token.trim().to_uppercase();
    match kw.param("_INSTANCE") {
        Some(i) => format!("{}.{t}", i.to_uppercase()),
        None => t,
    }
}

/// Är ytan en analytisk stel yta?
pub fn is_analytical(m: &FeModel, name: &str) -> bool {
    m.surfaces.get(&name.trim().to_uppercase()).is_some_and(|s| crate::inp::keywords::analytical::is_analytical_type(&s.kind))
}

// ---------------------------------------------------------------- profil

#[derive(Debug, Clone)]
enum Seg {
    Line { a: [f64; 2], b: [f64; 2] },
    /// Centrum, radie, startvinkel och (signerad) vinkeländring.
    Arc { c: [f64; 2], r: f64, t0: f64, dt: f64 },
    /// Andragradskurva genom p0 (t = 0), pm (t = ½) och p1 (t = 1).
    Parab { p0: [f64; 2], pm: [f64; 2], p1: [f64; 2] },
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len2(a: [f64; 2]) -> f64 {
    dot2(a, a).sqrt()
}

impl Seg {
    /// (punkt, enhetstangent i profilens riktning) vid parametern t ∈ [0, 1].
    fn eval(&self, t: f64) -> ([f64; 2], [f64; 2]) {
        match *self {
            Seg::Line { a, b } => {
                let d = sub2(b, a);
                let l = len2(d);
                ([a[0] + t * d[0], a[1] + t * d[1]], [d[0] / l, d[1] / l])
            }
            Seg::Arc { c, r, t0, dt } => {
                let th = t0 + t * dt;
                let sg = dt.signum();
                ([c[0] + r * th.cos(), c[1] + r * th.sin()], [-sg * th.sin(), sg * th.cos()])
            }
            Seg::Parab { p0, pm, p1 } => {
                let (l0, lm, l1) = (2.0 * (t - 0.5) * (t - 1.0), -4.0 * t * (t - 1.0), 2.0 * t * (t - 0.5));
                let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
                let p = [l0 * p0[0] + lm * pm[0] + l1 * p1[0], l0 * p0[1] + lm * pm[1] + l1 * p1[1]];
                let d = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
                let l = len2(d);
                (p, [d[0] / l, d[1] / l])
            }
        }
    }

    /// Närmaste punktens parameter (oklämd för linje och båge).
    fn closest_t(&self, p: [f64; 2]) -> f64 {
        match *self {
            Seg::Line { a, b } => {
                let d = sub2(b, a);
                dot2(sub2(p, a), d) / dot2(d, d)
            }
            Seg::Arc { c, t0, dt, .. } => {
                let phi = (p[1] - c[1]).atan2(p[0] - c[0]);
                let mut rel = phi - t0;
                // relativ vinkel i (−π, π] kring bågens mitt
                let mid = 0.5 * dt;
                while rel - mid > std::f64::consts::PI {
                    rel -= 2.0 * std::f64::consts::PI;
                }
                while rel - mid <= -std::f64::consts::PI {
                    rel += 2.0 * std::f64::consts::PI;
                }
                rel / dt
            }
            Seg::Parab { .. } => {
                // startvärde från en grov sökning, sedan Newton på (x(t) − p) · x'(t) = 0
                let mut best = (f64::INFINITY, 0.0);
                for k in 0..=20 {
                    let t = k as f64 / 20.0;
                    let (x, _) = self.eval(t);
                    let d = len2(sub2(x, p));
                    if d < best.0 {
                        best = (d, t);
                    }
                }
                let mut t = best.1;
                let Seg::Parab { p0, pm, p1 } = *self else { unreachable!() };
                let dd = [4.0 * p0[0] - 8.0 * pm[0] + 4.0 * p1[0], 4.0 * p0[1] - 8.0 * pm[1] + 4.0 * p1[1]];
                for _ in 0..50 {
                    let (x, _) = self.eval(t);
                    let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
                    let d = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
                    let r = sub2(x, p);
                    let g = dot2(r, d);
                    let h = dot2(d, d) + dot2(r, dd);
                    if h.abs() < 1e-300 {
                        break;
                    }
                    let st = g / h;
                    t -= st;
                    if st.abs() < 1e-15 {
                        break;
                    }
                }
                t
            }
        }
    }
}

/// Närmaste punkt på profilen: (punkt, utåtnormal, signerat avstånd).
/// `None` om punkten ligger utanför profilens ändar.
fn profile_closest(prof: &[Seg], p: [f64; 2]) -> Option<([f64; 2], [f64; 2], f64)> {
    let mut best: Option<(f64, usize, f64)> = None;
    for (i, s) in prof.iter().enumerate() {
        let t = s.closest_t(p).clamp(0.0, 1.0);
        let (q, _) = s.eval(t);
        let d = len2(sub2(p, q));
        if best.is_none_or(|b| d < b.0 - 1e-14 * (1.0 + d)) {
            best = Some((d, i, t));
        }
    }
    let (_, i, t) = best?;
    let tol = 1e-12;
    let (q, s) = prof[i].eval(t);
    let n = [-s[1], s[0]];
    let at_start = t <= tol;
    let at_end = t >= 1.0 - tol;
    if (at_start && i == 0) || (at_end && i + 1 == prof.len()) {
        // utanför profilens ände: bara om punkten ligger rakt ovanför/under änden
        let raw = prof[i].closest_t(p);
        if (at_start && raw < -1e-6) || (at_end && raw > 1.0 + 1e-6) {
            return None;
        }
    }
    if at_start || at_end {
        // hörn mellan två segment: normalen mot punkten (medelnormalens sida)
        let j = if at_start { i.wrapping_sub(1) } else { i + 1 };
        if let Some(o) = prof.get(j) {
            let (_, so) = o.eval(if at_start { 1.0 } else { 0.0 });
            // släta övergångar (samma tangent) behandlas som en vanlig punkt på ytan
            let smooth = dot2(so, s) > 1.0 - 1e-10;
            let navg = [n[0] - so[1], n[1] + so[0]];
            let d = sub2(p, q);
            let l = len2(d);
            if !smooth && l > 1e-6 * (1.0 + len2(q)) {
                let sg = if dot2(d, navg) >= 0.0 { 1.0 } else { -1.0 };
                let nn = [sg * d[0] / l, sg * d[1] / l];
                return Some((q, nn, sg * l));
            }
        }
    }
    let g = dot2(sub2(p, q), n);
    Some((q, n, g))
}

/// Normalens ändring längs profilen i punkten `p`: (κ', s) med
/// δn = κ' s (s · δp), där s är tangenten och κ' = κ/(1 + κ g) med κ > 0 för
/// en konvex yta (normalen bort från krökningscentrum). Hörn: κ' = 1/avståndet.
fn profile_curvature(prof: &[Seg], p: [f64; 2]) -> Option<(f64, [f64; 2])> {
    let (q, n, g) = profile_closest(prof, p)?;
    let s = [n[1], -n[0]];
    let mut best: Option<(f64, usize, f64)> = None;
    for (i, sg) in prof.iter().enumerate() {
        let t = sg.closest_t(p).clamp(0.0, 1.0);
        let (qq, _) = sg.eval(t);
        let d = len2(sub2(qq, q));
        if best.is_none_or(|b| d < b.0) {
            best = Some((d, i, t));
        }
    }
    let (_, i, t) = best?;
    let (_, si) = prof[i].eval(t);
    if dot2(si, s).abs() < 1.0 - 1e-9 {
        // hörn: normalen är riktningen mot punkten
        return Some((if g != 0.0 { 1.0 / g } else { 0.0 }, s));
    }
    let kappa = match prof[i] {
        Seg::Line { .. } => 0.0,
        Seg::Arc { r, dt, .. } => -dt.signum() / r,
        Seg::Parab { p0, pm, p1 } => {
            let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
            let x1 = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
            let x2 = [4.0 * p0[0] - 8.0 * pm[0] + 4.0 * p1[0], 4.0 * p0[1] - 8.0 * pm[1] + 4.0 * p1[1]];
            -(x1[0] * x2[1] - x1[1] * x2[0]) / len2(x1).powi(3)
        }
    };
    let den = 1.0 + kappa * g;
    Some((if den.abs() > 1e-12 { kappa / den } else { 0.0 }, s))
}

fn parse_profile(kw: &RawKeyword, recs: &[crate::fem::model::DataLine]) -> Result<Vec<Seg>> {
    let mut prof = Vec::new();
    let mut cur: Option<[f64; 2]> = None;
    for d in recs {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).collect();
        if f.iter().all(|s| s.is_empty()) {
            continue;
        }
        let word = f[0].to_uppercase();
        let v: Vec<f64> = f[1..].iter().filter(|s| !s.is_empty()).map(|s| parse_f64(s).ok_or(())).collect::<std::result::Result<_, _>>().or_else(|_| kw.err_at(d.line, "ogiltigt tal"))?;
        let need = |k: usize| -> Result<()> {
            if v.len() < k {
                return kw.err_at(d.line, format!("{word} kräver {k} koordinater"));
            }
            Ok(())
        };
        match word.as_str() {
            "START" => {
                need(2)?;
                if cur.is_some() {
                    return kw.err_at(d.line, "START får bara stå först");
                }
                cur = Some([v[0], v[1]]);
            }
            "LINE" | "CIRCL" | "PARAB" | "PARABOLA" => {
                let Some(a) = cur else { return kw.err_at(d.line, "profilen ska börja med START") };
                match word.as_str() {
                    "LINE" => {
                        need(2)?;
                        let b = [v[0], v[1]];
                        if len2(sub2(b, a)) == 0.0 {
                            return kw.err_at(d.line, "linjen har längden noll");
                        }
                        prof.push(Seg::Line { a, b });
                        cur = Some(b);
                    }
                    "CIRCL" => {
                        need(4)?;
                        let (b, c) = ([v[0], v[1]], [v[2], v[3]]);
                        let (r0, r1) = (len2(sub2(a, c)), len2(sub2(b, c)));
                        if r0 == 0.0 || (r0 - r1).abs() > 1e-6 * r0 {
                            return kw.err_at(d.line, "bågens start- och slutpunkt ligger inte på samma avstånd från centrum");
                        }
                        let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
                        let t1 = (b[1] - c[1]).atan2(b[0] - c[0]);
                        let mut dt = t1 - t0;
                        while dt > std::f64::consts::PI {
                            dt -= 2.0 * std::f64::consts::PI;
                        }
                        while dt <= -std::f64::consts::PI {
                            dt += 2.0 * std::f64::consts::PI;
                        }
                        if dt.abs() >= 179.74f64.to_radians() || dt == 0.0 {
                            return kw.err_at(d.line, "bågen ska vara kortare än 179,74° (dela den i två)");
                        }
                        prof.push(Seg::Arc { c, r: r0, t0, dt });
                        cur = Some(b);
                    }
                    _ => {
                        need(4)?;
                        let (pm, p1) = ([v[0], v[1]], [v[2], v[3]]);
                        prof.push(Seg::Parab { p0: a, pm, p1 });
                        cur = Some(p1);
                    }
                }
            }
            other => return kw.err_at(d.line, format!("okänt segment '{other}' (START, LINE, CIRCL, PARAB)")),
        }
    }
    if prof.is_empty() {
        return kw.err("den analytiska ytan saknar segment (LINE, CIRCL, PARAB)");
    }
    Ok(prof)
}

// ---------------------------------------------------------------- ytan

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// SEGMENTS eller CYLINDER: profilen i (x, y), z = ez.
    Swept,
    /// REVOLUTION: profilen i (r, z) kring ez.
    Revolution,
}

/// En analytisk yta i referensläget (globala koordinater).
#[derive(Debug, Clone)]
pub struct AnalyticalSurface {
    kind: Kind,
    a: [f64; 3],
    ex: [f64; 3],
    ey: [f64; 3],
    ez: [f64; 3],
    prof: Vec<Seg>,
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v).max(1e-300))
}

impl AnalyticalSurface {
    /// Läser ytan `name` (versaler, med instansprefix) ur modellens sparade *SURFACE.
    pub fn from_model(m: &FeModel, name: &str) -> Result<AnalyticalSurface> {
        let up = name.trim().to_uppercase();
        let Some(kw) = m.extra.iter().find(|k| k.name == "SURFACE" && k.param("NAME").is_some_and(|n| scoped(k, n) == up)) else {
            return fail(format!("den analytiska ytan {up} finns inte"));
        };
        if kw.param("FILLET RADIUS").and_then(parse_f64).is_some_and(|r| r != 0.0) {
            return kw.err("FILLET RADIUS stöds inte ännu");
        }
        let ty = kw.param_upper("TYPE").unwrap_or_default();
        let recs: Vec<crate::fem::model::DataLine> = kw.records().into_iter().filter(|d| d.fields.iter().any(|s| !s.trim().is_empty())).collect();
        let is_word = |d: &crate::fem::model::DataLine| d.fields.first().is_some_and(|s| s.trim().chars().next().is_some_and(|c| c.is_ascii_alphabetic()));
        let in_part = kw.has("_PART");
        let pts = |d: &crate::fem::model::DataLine, k: usize| -> Result<Vec<f64>> {
            let v = kw.floats(d)?;
            if v.len() < k {
                return kw.err_at(d.line, format!("raden ska ha {k} koordinater"));
            }
            Ok(v)
        };
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let (kind, a, ex, ey, ez, first) = match ty.as_str() {
            "SEGMENTS" => {
                if m.dim() != 2 {
                    return kw.err("TYPE=SEGMENTS gäller plana och axisymmetriska modeller (använd CYLINDER eller REVOLUTION i 3D)");
                }
                (Kind::Swept, [0.0; 3], x, y, z, 0)
            }
            "CYLINDER" => {
                if !recs.first().is_some_and(|d| !is_word(d)) {
                    if !in_part {
                        return kw.err("TYPE=CYLINDER kräver punkterna a, b och c på de två första dataraderna");
                    }
                    (Kind::Swept, [0.0; 3], x, y, z, 0)
                } else {
                    let v1 = pts(&recs[0], 6)?;
                    let Some(d2) = recs.get(1).filter(|d| !is_word(d)) else { return kw.err("punkten c (genereringsvektorn) saknas på andra dataraden") };
                    let v2 = pts(d2, 3)?;
                    let a = [v1[0], v1[1], v1[2]];
                    let ex = unit(sub([v1[3], v1[4], v1[5]], a));
                    let g = sub([v2[0], v2[1], v2[2]], a);
                    let g = sub(g, scale(ex, dot(g, ex)));
                    if norm(g) == 0.0 {
                        return kw.err("punkten c ligger på linjen ab");
                    }
                    // genereringsvektorn a → c är negativa lokala z-axeln
                    let ez = scale(unit(g), -1.0);
                    let ey = cross(ez, ex);
                    (Kind::Swept, a, ex, ey, ez, 2)
                }
            }
            "REVOLUTION" => {
                if !recs.first().is_some_and(|d| !is_word(d)) {
                    if !in_part {
                        return kw.err("TYPE=REVOLUTION kräver punkterna a och b på första dataraden");
                    }
                    // i en part: r längs x, symmetriaxeln längs y
                    (Kind::Revolution, [0.0; 3], x, [0.0, 0.0, -1.0], y, 0)
                } else {
                    let v1 = pts(&recs[0], 6)?;
                    let a = [v1[0], v1[1], v1[2]];
                    let ez = unit(sub([v1[3], v1[4], v1[5]], a));
                    let t = if ez[0].abs() < 0.9 { x } else { y };
                    let ex = unit(sub(t, scale(ez, dot(t, ez))));
                    let ey = cross(ez, ex);
                    (Kind::Revolution, a, ex, ey, ez, 1)
                }
            }
            other => return kw.err(format!("TYPE={other} stöds inte för analytiska ytor")),
        };
        let prof = parse_profile(kw, &recs[first..])?;
        let mut s = AnalyticalSurface { kind, a, ex, ey, ez, prof };
        if let Some(p) = placement_of(kw) {
            s.a = p.point(s.a);
            s.ex = p.direction(s.ex);
            s.ey = p.direction(s.ey);
            s.ez = p.direction(s.ez);
        }
        Ok(s)
    }

    /// Normalens variation för punkten `p` (referensläget): δn = Σ κ'ᵢ tᵢ (tᵢ · δp)
    /// med tangentriktningarna tᵢ (profilens riktning och, för REVOLUTION,
    /// omkretsriktningen med κ' = n_r / r).
    pub fn curvatures(&self, p: [f64; 3]) -> Vec<(f64, [f64; 3])> {
        let d = sub(p, self.a);
        let zc = dot(d, self.ez);
        match self.kind {
            Kind::Swept => {
                let Some((k, s)) = profile_curvature(&self.prof, [dot(d, self.ex), dot(d, self.ey)]) else { return Vec::new() };
                vec![(k, [0, 1, 2].map(|c| s[0] * self.ex[c] + s[1] * self.ey[c]))]
            }
            Kind::Revolution => {
                let rad = sub(d, scale(self.ez, zc));
                let r = norm(rad);
                if r <= 1e-14 * (1.0 + norm(d)) {
                    return Vec::new();
                }
                let er = scale(rad, 1.0 / r);
                let Some((k, s)) = profile_curvature(&self.prof, [r, zc]) else { return Vec::new() };
                let Some((_, n, _)) = profile_closest(&self.prof, [r, zc]) else { return Vec::new() };
                let et = cross(self.ez, er);
                vec![(k, [0, 1, 2].map(|c| s[0] * er[c] + s[1] * self.ez[c])), (n[0] / r, et)]
            }
        }
    }

    /// Närmaste punkt och utåtnormal för punkten `p` (referensläget):
    /// (punkt, normal, signerat avstånd).
    pub fn closest(&self, p: [f64; 3]) -> Option<([f64; 3], [f64; 3], f64)> {
        let d = sub(p, self.a);
        let zc = dot(d, self.ez);
        match self.kind {
            Kind::Swept => {
                let (xl, yl) = (dot(d, self.ex), dot(d, self.ey));
                let (q, n, g) = profile_closest(&self.prof, [xl, yl])?;
                let q3 = [0, 1, 2].map(|k| self.a[k] + q[0] * self.ex[k] + q[1] * self.ey[k] + zc * self.ez[k]);
                let n3 = [0, 1, 2].map(|k| n[0] * self.ex[k] + n[1] * self.ey[k]);
                Some((q3, n3, g))
            }
            Kind::Revolution => {
                let rad = sub(d, scale(self.ez, zc));
                let r = norm(rad);
                let er = if r > 1e-14 * (1.0 + norm(d)) { scale(rad, 1.0 / r) } else { self.ex };
                let (q, n, g) = profile_closest(&self.prof, [r, zc])?;
                let q3 = [0, 1, 2].map(|k| self.a[k] + q[0] * er[k] + q[1] * self.ez[k]);
                let n3 = [0, 1, 2].map(|k| n[0] * er[k] + n[1] * self.ez[k]);
                Some((q3, n3, g))
            }
        }
    }
}

// ---------------------------------------------------------------- kontakten

#[derive(Debug, Clone, Copy, Default)]
struct NodeState {
    p: f64,
    gap: f64,
    /// Elastisk glidning i kroppens system.
    slip: [f64; 3],
    /// Total glidning (rumslig, för CSLIP).
    total_slip: [f64; 3],
    shear: [f64; 3],
    normal: [f64; 3],
    contact: bool,
    /// Slavnodens läge i kroppens system.
    pb: [f64; 3],
    /// CPRESS (kontaktkraft / aktuell area).
    cpress: f64,
    /// Antal utvärderingar som noden varit sluten i följd (kontaktstatus).
    age: u32,
}

#[derive(Debug, Default)]
struct State {
    committed: Vec<NodeState>,
    trial: Vec<NodeState>,
    /// Största kontakttrycket i senaste utvärderingen (för öppningstoleransen).
    pmax: f64,
}

/// Area per slavnod (axisymmetriskt hela varvet, 2π r, som elementens
/// krafter) i referensläget eller, med `u`, i aktuellt läge.
fn slave_areas(m: &FeModel, facets: &[Facet], node_weights: &[(usize, f64)], nodes: &[usize], u: Option<&[NodeVec]>) -> Vec<f64> {
    let idx = |n: usize| nodes.binary_search(&n).unwrap();
    let mut area = vec![0.0; nodes.len()];
    for f in facets {
        let x = geom::coords(m, f, u);
        let axi = m.elements[f.elem].etype.family() == crate::elements::Family::Axisymmetric;
        for (xi, w) in geom::gauss(f.dim, x.len(), 3) {
            let fp = geom::eval(f, &x, xi);
            let wr = if axi { 2.0 * std::f64::consts::PI * fp.x[0] } else { 1.0 };
            for k in 0..x.len() {
                area[idx(f.nodes[k])] += fp.n[k] * w * fp.jac * wr;
            }
        }
    }
    for &(n, w) in node_weights {
        area[idx(n)] += if w > 0.0 { w } else { 1.0 };
    }
    area
}

pub struct AnalyticalContact {
    name: String,
    nodes: Vec<usize>,
    /// Slavnodernas area i referensläget.
    area: Vec<f64>,
    facets: Vec<Facet>,
    node_weights: Vec<(usize, f64)>,
    /// Hård kontakt med direkt metod (kraften ur straffet, CPRESS = kraft/area).
    direct: bool,
    k_hard: Vec<f64>,
    /// Slavsidans elementstyvhet per slavnod (gräns för krökningstermen i tangenten).
    k_elem: Vec<f64>,
    surf: AnalyticalSurface,
    refn: usize,
    xr0: [f64; 3],
    props: PairProps,
    gamma: f64,
    dim: usize,
    /// SMALL SLIDING: tangentplanet (punkt, normal) i kroppens system per slavnod.
    anchors: Option<Vec<Option<([f64; 3], [f64; 3])>>>,
    state: Mutex<State>,
}

/// Referensnoden för den stela kropp som äger den analytiska ytan `surf`.
fn rigid_ref(m: &FeModel, kw: &RawKeyword, surf: &str) -> Result<usize> {
    let up = surf.trim().to_uppercase();
    for rb in m.extra.iter().filter(|k| k.name == "RIGID BODY") {
        let Some(s) = rb.param("ANALYTICAL SURFACE") else { continue };
        if scoped(rb, s) != up {
            continue;
        }
        let r = crate::inp::required(rb, "REF NODE")?;
        let v = geom::node_ref(m, rb, &scoped(rb, r))?;
        if v.len() != 1 {
            return rb.err(format!("REF NODE={r} ska vara en enda nod"));
        }
        return Ok(v[0]);
    }
    kw.err(format!("den analytiska ytan {up} ingår inte i någon *RIGID BODY, ANALYTICAL SURFACE="))
}

fn build(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let mut out: Vec<Box<dyn Interaction>> = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.len() < 2 || !is_analytical(m, f[1]) {
            continue;
        }
        if is_analytical(m, f[0]) {
            return kw.err_at(d.line, "en analytisk yta kan bara vara masteryta");
        }
        if kw.param_upper("TYPE").is_some_and(|t| t == "SURFACE TO SURFACE") {
            return kw.err_at(d.line, "TYPE=SURFACE TO SURFACE mot en analytisk yta stöds inte ännu (använd NODE TO SURFACE)");
        }
        if m.extra.iter().any(|c| c.name == "CLEARANCE" && c.param_upper("MASTER").is_some_and(|x| x == f[1].to_uppercase())) {
            return kw.err_at(d.line, "*CLEARANCE mot en analytisk yta stöds inte ännu");
        }
        let ip = crate::inp::required(kw, "INTERACTION")?;
        let props = PairProps::read(m, kw, ip)?;
        let name = format!("*CONTACT PAIR {}/{} ({})", f[0], f[1], kw.location());
        out.push(Box::new(AnalyticalContact::new(m, kw, name, f[0], f[1], props, kw.has("SMALL SLIDING"))?));
    }
    Ok(out)
}

impl AnalyticalContact {
    fn new(m: &FeModel, kw: &RawKeyword, name: String, slave: &str, master: &str, props: PairProps, small: bool) -> Result<AnalyticalContact> {
        let surf = AnalyticalSurface::from_model(m, master)?;
        let refn = rigid_ref(m, kw, master)?;
        let ss = geom::surface(m, kw, slave)?;
        let dim = m.dim();
        let mut nodes: Vec<usize>;
        let mut facets: Vec<Facet> = Vec::new();
        if ss.kind == "NODE" {
            nodes = ss.nodes.iter().map(|p| p.0).collect();
        } else {
            facets = geom::facets(m, ss, &name)?;
            nodes = facets.iter().flat_map(|f| f.nodes.iter().copied()).collect();
        }
        nodes.sort_unstable();
        nodes.dedup();
        let node_weights: Vec<(usize, f64)> = if ss.kind == "NODE" { ss.nodes.clone() } else { Vec::new() };
        let area = slave_areas(m, &facets, &node_weights, &nodes, None);
        let sizes: Vec<f64> = facets.iter().map(|f| contact::facet_length(f, &geom::coords(m, f, None))).collect();
        // straffkoefficient per slavnod
        let mut k_hard = Vec::with_capacity(nodes.len());
        let mut k_elem = Vec::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            // slavsidans elementstyvhet i noden
            let mut s = 0.0;
            let mut c = 0usize;
            for (e, el) in m.elements.iter().enumerate() {
                if el.nodes.contains(&n) && !crate::elements::rigid::is_rigid(el.etype) {
                    s += contact::representative_stiffness(m, e)?;
                    c += 1;
                }
            }
            let ks = if c > 0 { s / c as f64 } else { 1.0 };
            k_elem.push(ks);
            let k = match props.penalty() {
                Some((Some(k), sc)) => k * sc,
                Some((None, sc)) => {
                    let v: Vec<f64> = facets.iter().filter(|f| f.nodes.contains(&n)).map(|f| contact::default_penalty(m, f)).collect::<Result<_>>()?;
                    sc * v.iter().sum::<f64>() / v.len().max(1) as f64
                }
                // direkt: mycket styv straffkoefficient från slavsidans element
                None => HARD_FACTOR_ANALYTICAL * ks / area[i].max(1e-300),
            };
            k_hard.push(k);
        }
        let lc = if sizes.is_empty() { 1.0 } else { sizes.iter().sum::<f64>() / sizes.len() as f64 };
        let xr0 = m.nodes[refn];
        let anchors = small.then(|| nodes.iter().map(|&n| surf.closest(m.nodes[n]).map(|(q, nn, _)| (q, nn))).collect());
        let committed: Vec<NodeState> = nodes.iter().map(|&n| NodeState { pb: m.nodes[n], ..Default::default() }).collect();
        Ok(AnalyticalContact {
            name,
            area,
            facets,
            node_weights,
            direct: props.penalty().is_none() && props.is_hard(),
            k_hard,
            k_elem,
            surf,
            refn,
            xr0,
            props,
            gamma: props.slip_tol() * lc,
            dim,
            anchors,
            state: Mutex::new(State { trial: committed.clone(), committed, pmax: 0.0 }),
            nodes,
        })
    }

    fn rotation(&self, u: &[NodeVec]) -> [[f64; 3]; 3] {
        let ur = &u[self.refn];
        let th = if self.dim == 2 { [0.0, 0.0, ur[5]] } else { [ur[3], ur[4], ur[5]] };
        tensor::exp_so3(th)
    }

    fn ref_dofs(&self) -> Vec<Dof> {
        let mut v = geom::trans_dofs(self.dim).to_vec();
        v.extend(geom::rot_dofs(self.dim));
        v
    }
}

impl Interaction for AnalyticalContact {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        self.ref_dofs().into_iter().map(|d| (self.refn, d)).collect()
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.nodes.iter().map(|&n| vec![n, self.refn]).collect()
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], mut k: Option<&mut dyn TangentSink>) -> Result<()> {
        let m = ctx.model;
        let dim = self.dim;
        let rot = self.rotation(u);
        let rt = tensor::transpose(&rot);
        let ur = u[self.refn];
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        let mut pmax = 0.0f64;
        let area_cur = if self.anchors.is_none() { slave_areas(m, &self.facets, &self.node_weights, &self.nodes, Some(u)) } else { self.area.clone() };
        for (i, &sn) in self.nodes.iter().enumerate() {
            let old = st.committed[i];
            let prev_eval = st.trial[i];
            // slavnoden i kroppens referensläge: p_b = X_s + d med
            // d = (Rᵀ − I)(X_s − X_R) + Rᵀ(u_s − u_R); gapet räknas som
            // n · (X_s − q) + n · d så att avrundningen i de stora koordinaterna
            // inte ger brus i straffkraften mellan iterationerna
            let xs0 = m.nodes[sn];
            let r0 = sub(xs0, self.xr0);
            let w = sub([u[sn][0], u[sn][1], u[sn][2]], [ur[0], ur[1], ur[2]]);
            let rr0 = tensor::mv(&rt, r0);
            let rw = tensor::mv(&rt, w);
            let dpb = [rr0[0] - r0[0] + rw[0], rr0[1] - r0[1] + rw[1], rr0[2] - r0[2] + rw[2]];
            let pb = [xs0[0] + dpb[0], xs0[1] + dpb[1], xs0[2] + dpb[2]];
            let mut ns = NodeState { total_slip: old.total_slip, pb, ..Default::default() };
            let proj = match &self.anchors {
                Some(a) => a[i].map(|(q, n)| (q, n)),
                None => self.surf.closest(pb).map(|(q, n, _)| (q, n)),
            };
            let Some((qb, nb)) = proj else {
                st.trial[i] = ns;
                continue;
            };
            let g = dot(sub(xs0, qb), nb) + dot(dpb, nb);
            let n = tensor::mv(&rot, nb);
            let a = tensor::mv(&rot, sub(qb, self.xr0));
            ns.gap = g;
            ns.normal = n;
            if crate::settings::with(|s| s.contact_debug) {
                eprintln!("analytisk kontakt {} nod {}: g={g:.6e} n={n:?} a={a:?}", self.name, m.node_name(sn));
            }
            let h = -g;
            let kh = self.k_hard[i];
            // hård kontakt med kontaktstatus: en sluten nod öppnas
            // först vid dragtryck (utöver en liten tolerans), en öppen sluts vid
            // överlappning; annars pendlar noder vid kontaktzonens kant mellan
            // öppen (överlappning) och sluten (drag) från iteration till iteration
            let (p, dp, closed) = if self.props.is_hard() {
                let was = st.trial[i].contact;
                let age = st.trial[i].age;
                // en nod som just har slutits hålls sluten en iteration till:
                // gapets linjärisering (krökt yta, tangentiell rörelse) ger annars
                // ett litet fel som med den styva straffkoefficienten ser ut som drag
                let closed = if was { age < 2 || kh * h >= -1e-8 * st.pmax } else { h > 0.0 };
                ns.age = if closed && was { age.saturating_add(1) } else { 0 };
                if closed != was && crate::settings::with(|s| s.contact_debug) {
                    eprintln!("  status {}: {} h={h:.4e} x={pb:?}", m.node_name(sn), if closed { "sluten" } else { "öppen" });
                }
                if closed { (kh * h, kh, true) } else { (0.0, 0.0, false) }
            } else {
                let (p, dp) = self.props.pressure(h, kh);
                (p, dp, self.props.active(h))
            };
            pmax = pmax.max(p);
            ns.p = p;
            ns.contact = closed;
            // friktion i kroppens system
            let mut tau_b = [0.0; 3];
            let mut kt_b: Option<[[f64; 3]; 3]> = None;
            let dpb = sub(pb, old.pb);
            let dn = dot(dpb, nb);
            let ds = [dpb[0] - dn * nb[0], dpb[1] - dn * nb[1], dpb[2] - dn * nb[2]];
            let mu = self.props.mu();
            // friktion bara för noder som var slutna i föregående utvärdering: en nod
            // som just slöts (ofta med stor överlappning från extrapolationen) får
            // annars en orimlig friktionskraft μp
            let fric = mu > 0.0 && ns.contact && p > 0.0 && prev_eval.contact && prev_eval.age >= 1;
            if ns.contact && !fric {
                let d = tensor::mv(&rot, ds);
                for c in 0..3 {
                    ns.total_slip[c] += d[c];
                }
            }
            if fric {
                let eo = old.slip;
                let eon = dot(eo, nb);
                let e_tr = [eo[0] - eon * nb[0] + ds[0], eo[1] - eon * nb[1] + ds[1], eo[2] - eon * nb[2] + ds[2]];
                let e_len = norm(e_tr);
                let mu_p = mu * p;
                let gam = self.gamma.max(1e-300);
                let mut pt = [[0.0; 3]; 3];
                for r in 0..3 {
                    for c in 0..3 {
                        pt[r][c] = (if r == c { 1.0 } else { 0.0 }) - nb[r] * nb[c];
                    }
                }
                if dim == 2 {
                    pt[2] = [0.0; 3];
                    pt[0][2] = 0.0;
                    pt[1][2] = 0.0;
                }
                if e_len <= gam {
                    tau_b = scale(e_tr, mu_p / gam);
                    ns.slip = e_tr;
                    kt_b = Some(pt.map(|row| row.map(|v| v * mu_p / gam)));
                } else {
                    let eh = scale(e_tr, 1.0 / e_len);
                    tau_b = scale(eh, mu_p);
                    ns.slip = scale(eh, gam);
                    let f = mu_p / e_len;
                    let mut kk = [[0.0; 3]; 3];
                    for r in 0..3 {
                        for c in 0..3 {
                            kk[r][c] = f * (pt[r][c] - eh[r] * eh[c]);
                        }
                    }
                    // fastnod i föregående utvärdering (eller glidning åt andra
                    // hållet): fastlägets tangent. Glidningens tangent saknar styvhet
                    // i glidriktningen, så Newton pendlar annars mellan glidning åt
                    // båda hållen när noden egentligen fastnar.
                    let prev = prev_eval;
                    let stuck_before = !prev.contact || norm(prev.slip) < gam * (1.0 - 1e-9) || dot(prev.slip, eh) < 0.0;
                    kt_b = Some(if stuck_before { pt.map(|row| row.map(|v| v * mu_p / gam)) } else { kk });
                }
                let dslip = sub(ds, sub(ns.slip, [eo[0] - eon * nb[0], eo[1] - eon * nb[1], eo[2] - eon * nb[2]]));
                let d = tensor::mv(&rot, dslip);
                for c in 0..3 {
                    ns.total_slip[c] = old.total_slip[c] + d[c];
                }
            }
            let tau = tensor::mv(&rot, tau_b);
            if crate::settings::with(|s| s.contact_debug) {
                eprintln!("  friktion {}: p={p:.4e} ds={ds:?} tau={tau:?} uR={:?} us={:?}", m.node_name(sn), &ur[..3], &u[sn][..3]);
            }
            ns.shear = tau;
            st.trial[i] = ns;
            if !ns.contact && p == 0.0 {
                continue;
            }
            // kraften: direkt metod ur straffet (k = f · k_e / A₀), annars p · A i
            // aktuellt läge; CPRESS = kraft / aktuell area (finite sliding)
            let area = if self.direct { self.area[i] } else { area_cur[i] };
            st.trial[i].cpress = p * area / area_cur[i].max(1e-300);
            let tr = [-p * n[0] + tau[0], -p * n[1] + tau[1], -p * n[2] + tau[2]];
            let r = self.refn;
            for d in 0..dim {
                fint[sn][d] += area * tr[d];
                fint[r][d] -= area * tr[d];
            }
            let mom = cross(a, tr);
            if dim == 2 {
                fint[r][5] -= area * mom[2];
            } else {
                for d in 0..3 {
                    fint[r][3 + d] -= area * mom[d];
                }
            }
            let Some(k) = k.as_deref_mut() else { continue };
            // D = p' n nᵀ + R K_t Rᵀ (rumsligt)
            let mut dmat = [[0.0; 3]; 3];
            for r1 in 0..3 {
                for c in 0..3 {
                    dmat[r1][c] = dp * n[r1] * n[c];
                }
            }
            // ytans krökning (finite sliding): δ(−p n) = −p κ' t tᵀ δx. Bara för noder som
            // var slutna redan i föregående utvärdering (en nod som just slöts kan ha
            // ett stort tillfälligt straffryck, och termen är destabiliserande för
            // konvexa ytor)
            // termen begränsas till en fjärdedel av elementstyvheten i noden (ett
            // stort tillfälligt straffryck i en iteration gör annars tangenten indefinit)
            if self.anchors.is_none() && p > 0.0 && prev_eval.contact {
                for (kap, tb) in self.surf.curvatures(pb) {
                    let t = tensor::mv(&rot, tb);
                    let pg = p.min(0.25 * self.k_elem[i] / (area * kap.abs()).max(1e-300));
                    for r1 in 0..3 {
                        for c in 0..3 {
                            dmat[r1][c] -= pg * kap * t[r1] * t[c];
                        }
                    }
                }
            }
            if let Some(kb) = kt_b {
                let ks = tensor::mul(&tensor::mul(&rot, &kb), &rt);
                for r1 in 0..3 {
                    for c in 0..3 {
                        dmat[r1][c] += ks[r1][c];
                    }
                }
            }
            // G: relativ förskjutning = δu_s − δu_R + a × δw
            let mut cols: Vec<(usize, Dof, [f64; 3])> = Vec::new();
            for d in 0..dim {
                let mut e = [0.0; 3];
                e[d] = 1.0;
                cols.push((sn, U1 + d as Dof, e));
                cols.push((r, U1 + d as Dof, scale(e, -1.0)));
            }
            if dim == 2 {
                cols.push((r, UR3, cross(a, [0.0, 0.0, 1.0])));
            } else {
                for j in 0..3 {
                    let mut e = [0.0; 3];
                    e[j] = 1.0;
                    cols.push((r, UR1 + j as Dof, cross(a, e)));
                }
            }
            for (na, da, ga) in &cols {
                for (nb2, db, gb) in &cols {
                    let mut v = 0.0;
                    for r1 in 0..3 {
                        for c in 0..3 {
                            v += ga[r1] * dmat[r1][c] * gb[c];
                        }
                    }
                    let v = area * v;
                    if v != 0.0 {
                        k.add(*na, *da, *nb2, *db, v);
                    }
                }
            }
        }
        st.pmax = pmax;
        Ok(())
    }

    fn commit(&self) {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        st.committed.clone_from(&st.trial);
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let st = self.state.lock().unwrap();
        let mut cp = NodeOutput { name: "CPRESS".into(), labels: vec!["CPRESS".into()], values: Vec::new() };
        let mut co = NodeOutput { name: "COPEN".into(), labels: vec!["COPEN".into()], values: Vec::new() };
        let ncomp = self.dim - 1;
        let mut sh: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSHEAR{k}"), labels: vec![format!("CSHEAR{k}")], values: Vec::new() }).collect();
        let mut sl: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSLIP{k}"), labels: vec![format!("CSLIP{k}")], values: Vec::new() }).collect();
        for (i, &n) in self.nodes.iter().enumerate() {
            let s = st.committed[i];
            cp.values.push((n, vec![s.cpress.max(0.0)]));
            co.values.push((n, vec![s.gap]));
            let t = contact::output_tangents(s.normal, self.dim);
            for k in 0..ncomp {
                sh[k].values.push((n, vec![dot(s.shear, t[k])]));
                sl[k].values.push((n, vec![dot(s.total_slip, t[k])]));
            }
        }
        let mut out = vec![cp, co];
        out.extend(sh);
        out.extend(sl);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profilens_normal_och_bage() {
        // linje från (1, 0) till (−1, 0): normalen = riktningen vriden 90° moturs = (0, −1)
        let p = vec![Seg::Line { a: [1.0, 0.0], b: [-1.0, 0.0] }];
        let (_, n, g) = profile_closest(&p, [0.2, 0.5]).unwrap();
        assert!((n[1] + 1.0).abs() < 1e-14 && (g + 0.5).abs() < 1e-14);
        // kvartscirkel moturs kring origo med radien 2: normalen mot centrum
        let a = Seg::Arc { c: [0.0, 0.0], r: 2.0, t0: 0.0, dt: std::f64::consts::FRAC_PI_2 };
        let (q, n, g) = profile_closest(&[a], [3.0, 3.0]).unwrap();
        let s = 2.0 / 2f64.sqrt();
        assert!((q[0] - s).abs() < 1e-12 && (q[1] - s).abs() < 1e-12);
        assert!((n[0] + 1.0 / 2f64.sqrt()).abs() < 1e-12 && (g + (18f64.sqrt() - 2.0)).abs() < 1e-12);
    }
}
