//! Utökningar av kontakten i implicit analys (inkluderas med `#[path]` från
//! `contact.rs`):
//!
//! - *FRICTION, ANISOTROPIC (3D): friktionskoefficienterna μ₁, μ₂ i de lokala
//!   tangentriktningarna t₁, t₂ (samma som för CSHEAR1/2). Elastisk glidning
//!   med styvheten μᵢ p/γ i riktning i; glidgränsen är ellipsen
//!   (τ₁/μ₁)² + (τ₂/μ₂)² = p², och i glidning är τᵢ = μᵢ p êᵢ med ê
//!   riktningen för den elastiska glidningens prediktor (radiell återföring i
//!   de normerade spänningarna τᵢ/μᵢ, så att isotrop friktion är specialfallet
//!   μ₁ = μ₂).

use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

/// Självkontakt för ytan `name` (*CONTACT INCLUSIONS med en yta eller
/// ALL EXTERIOR för en kropp som kan nå sig själv): ytan är både slav och
/// master, med vikten ½ eftersom varje kontaktställe räknas från båda
/// sidorna (balanserad master–slav som i general contact). Masterytor
/// vars element innehåller slavpunktens noder hoppas över, så att en nod inte
/// får kontakt med sina egna element.
pub(super) fn self_pair(m: &FeModel, kw: &RawKeyword, name: &str, per_pair: &[(String, String, String)], p: super::Props) -> Result<super::ContactPair> {
    let up = name.to_uppercase();
    let props = match per_pair.iter().find(|(a, b, _)| *a == up && *b == up) {
        Some((_, _, n)) => super::props(m, kw, n)?,
        None => p,
    };
    let label = format!("*CONTACT {name} (självkontakt, {})", kw.location());
    let mut cp = super::ContactPair::new(m, kw, label, name, name, props, true, false, 0.5)?;
    cp.make_self(m);
    Ok(cp)
}

/// Kroppen (dess yttre ytor) är inte konvex i referensläget, så att den kan
/// komma i kontakt med sig själv: någon nod ligger framför någon ytas plan.
pub(super) fn can_self_contact(m: &FeModel, faces: &[(usize, String)]) -> bool {
    let fs: Vec<super::Facet> = faces
        .iter()
        .filter_map(|(e, f)| {
            let el = &m.elements[*e];
            let k: usize = f.trim_start_matches('S').parse().ok()?;
            let face = el.etype.faces().get(k.checked_sub(1)?)?;
            let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            Some(super::Facet { elem: *e, nodes: face.iter().map(|&q| el.nodes[q]).collect(), inside: crate::elements::common::centroid(&xs), dim: el.etype.dim(), thick: 1.0 })
        })
        .collect();
    let mut nodes: Vec<usize> = fs.iter().flat_map(|f| f.nodes.iter().copied()).collect();
    nodes.sort_unstable();
    nodes.dedup();
    for f in &fs {
        let x = super::geom::coords(m, f, None);
        let fp = super::geom::eval(f, &x, [0.0, 0.0]);
        let size = super::facet_size(f, &x);
        for &n in &nodes {
            let d = sub(m.nodes[n], fp.x);
            if dot(d, fp.normal) > 1e-6 * size {
                return true;
            }
        }
    }
    false
}

/// Justering av slavnoder i början av analysen (töjningsfritt):
///
/// - *CONTACT PAIR, ADJUST=a: slavnoder med gapet g ≤ a (även överlappande)
///   flyttas till masterytan längs dess normal; ADJUST=nodset: nodsetets
///   slavnoder flyttas till masterytan.
/// - *CONTACT INITIALIZATION DATA (via *CONTACT INITIALIZATION ASSIGNMENT i
///   *CONTACT): slavnoder med −SEARCH BELOW ≤ g ≤ SEARCH ABOVE flyttas så att
///   gapet blir INITIAL CLEARANCE (standard 0; utan SEARCH ABOVE bara
///   överlappande noder). Ytparet tolkas som för general contact (första ytan
///   slav). INITIAL CLEARANCE brukar inte flytta noderna; här
///   flyttas de till det angivna gapet (samma kontaktgap, något ändrad geometri).
///
/// Masterytan är den närmaste masterytan i referensläget (projektionen inom
/// ytan); nodernas nya lägen blir referenslägen (U räknas därifrån).
pub(super) fn adjust(m: &mut FeModel) -> Result<()> {
    let mut moves: Vec<(usize, [f64; 3])> = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT PAIR") {
        let Some(a) = kw.param("ADJUST") else { continue };
        let a = a.trim();
        let (limit, set): (Option<f64>, Option<Vec<usize>>) = match crate::fem::model::parse_f64(a) {
            Some(v) => (Some(v), None),
            None if a.eq_ignore_ascii_case("NO") => continue,
            None => (None, Some(super::geom::node_ref(m, kw, a)?)),
        };
        if limit == Some(0.0) {
            continue;
        }
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if f.len() < 2 || super::super::analytical::is_analytical(m, f[1]) {
                continue;
            }
            for (n, g, nm) in gaps(m, kw, f[0], f[1])? {
                let take = match (&set, limit) {
                    (Some(s), _) => s.contains(&n),
                    (None, Some(l)) => g <= l,
                    _ => false,
                };
                if take {
                    moves.push((n, [-g * nm[0], -g * nm[1], -g * nm[2]]));
                }
            }
        }
    }
    // general contact: *CONTACT INITIALIZATION ASSIGNMENT → *CONTACT INITIALIZATION DATA
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT INITIALIZATION ASSIGNMENT") {
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).collect();
            if f.len() < 3 || f[0].is_empty() || f[1].is_empty() || f[2].is_empty() {
                return kw.err_at(d.line, "skriv 'yta, yta, initieringsnamn' (två ytor krävs)");
            }
            let up = f[2].to_uppercase();
            let Some(init) = m.extra.iter().find(|k| k.name == "CONTACT INITIALIZATION DATA" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
                return kw.err_at(d.line, format!("*CONTACT INITIALIZATION DATA, NAME={} finns inte", f[2]));
            };
            if init.has("INTERFERENCE FIT") {
                return init.err("INTERFERENCE FIT stöds inte ännu");
            }
            let num = |k: &str| -> Result<Option<f64>> {
                match init.param(k) {
                    Some(v) => crate::fem::model::parse_f64(v).map(Some).ok_or_else(|| crate::model::FemError(format!("{}: {k}= ska vara ett tal", init.location()))),
                    None => Ok(None),
                }
            };
            let c = num("INITIAL CLEARANCE")?.unwrap_or(0.0);
            let above = num("SEARCH ABOVE")?.unwrap_or(0.0).max(c);
            let below = num("SEARCH BELOW")?;
            for (n, g, nm) in gaps(m, kw, f[0], f[1])? {
                let lo = below.map_or(f64::NEG_INFINITY, |b| -b);
                if g >= lo && (g <= above || g < c) && g != c {
                    let s = c - g;
                    moves.push((n, [s * nm[0], s * nm[1], s * nm[2]]));
                }
            }
        }
    }
    for (n, d) in moves {
        for k in 0..3 {
            m.nodes[n][k] += d[k];
        }
    }
    Ok(())
}

/// (slavnod, gap, masternormal) för slavytans noder mot närmaste masteryta
/// (projektionen inom ytan) i referensläget.
fn gaps(m: &FeModel, kw: &RawKeyword, slave: &str, master: &str) -> Result<Vec<(usize, f64, [f64; 3])>> {
    let ss = super::geom::surface(m, kw, slave)?;
    let ms = super::geom::surface(m, kw, master)?;
    let nodes = super::geom::surface_nodes(m, ss)?;
    let mf = super::geom::facets(m, ms, "masteryta")?;
    let mut out = Vec::new();
    for n in nodes {
        let p = m.nodes[n];
        let mut best: Option<(f64, f64, [f64; 3])> = None;
        for f in &mf {
            if f.nodes.contains(&n) {
                continue;
            }
            let x = super::geom::coords(m, f, None);
            let (xi, dist, inside) = super::geom::project(f, &x, p);
            if !inside {
                continue;
            }
            let fp = super::geom::eval(f, &x, xi);
            if best.is_none_or(|b| dist < b.0) {
                best = Some((dist, dot(sub(p, fp.x), fp.normal), fp.normal));
            }
        }
        if let Some((_, g, nm)) = best {
            out.push((n, g, nm));
        }
    }
    Ok(out)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// μ₁, μ₂ från dataraden till *FRICTION, ANISOTROPIC.
pub(super) fn anisotropic(sub: &RawKeyword) -> Result<(f64, f64)> {
    if sub.data.len() > 1 {
        return sub.err("tabulerad anisotrop friktion stöds inte ännu (en rad med μ₁, μ₂)");
    }
    let v = sub.first_floats()?;
    match v.as_slice() {
        [m1, m2, ..] if *m1 >= 0.0 && *m2 >= 0.0 => Ok((*m1, *m2)),
        [m1] => Ok((*m1, *m1)),
        _ => sub.err("*FRICTION, ANISOTROPIC kräver μ₁, μ₂"),
    }
}

/// Friktionens svar för en slavnod.
pub(super) struct Aniso {
    pub tau: [f64; 3],
    pub dtau_dp: [f64; 3],
    pub slip: [f64; 3],
    /// ∂τ/∂e (osymmetrisk när μ₁ ≠ μ₂; den symmetriska delen används i symmetriska lösare).
    pub kt: [[f64; 3]; 3],
}

/// Anisotrop friktion för den elastiska glidningens prediktor `e_tr`
/// (i tangentplanet), normalen `n`, trycket `p` och den kritiska elastiska
/// glidningen `gam`.
pub(super) fn aniso_friction(e_tr: [f64; 3], n: [f64; 3], p: f64, mu1: f64, mu2: f64, gam: f64) -> Aniso {
    let t = super::local_tangents(n, 3);
    let mu = [mu1, mu2];
    let e = [dot(e_tr, t[0]), dot(e_tr, t[1])];
    let e_len = (e[0] * e[0] + e[1] * e[1]).sqrt();
    let mut out = Aniso { tau: [0.0; 3], dtau_dp: [0.0; 3], slip: e_tr, kt: [[0.0; 3]; 3] };
    // dτ_i/de_j i lokala riktningar
    let mut d = [[0.0; 2]; 2];
    let mut tl = [0.0; 2];
    let mut dpl = [0.0; 2];
    if e_len <= gam {
        for i in 0..2 {
            tl[i] = mu[i] * p * e[i] / gam;
            dpl[i] = mu[i] * e[i] / gam;
            d[i][i] = mu[i] * p / gam;
        }
    } else {
        let eh = [e[0] / e_len, e[1] / e_len];
        for i in 0..2 {
            tl[i] = mu[i] * p * eh[i];
            dpl[i] = mu[i] * eh[i];
            for j in 0..2 {
                d[i][j] = mu[i] * p * ((if i == j { 1.0 } else { 0.0 }) - eh[i] * eh[j]) / e_len;
            }
        }
        out.slip = std::array::from_fn(|c| gam * (eh[0] * t[0][c] + eh[1] * t[1][c]));
    }
    for c in 0..3 {
        out.tau[c] = tl[0] * t[0][c] + tl[1] * t[1][c];
        out.dtau_dp[c] = dpl[0] * t[0][c] + dpl[1] * t[1][c];
    }
    for a in 0..3 {
        for b in 0..3 {
            let mut v = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    v += t[i][a] * d[i][j] * t[j][b];
                }
            }
            out.kt[a][b] = v;
        }
    }
    out
}

/// Symmetrisk och snedsymmetrisk del av en 3×3-matris.
pub(super) fn split(k: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let s = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (k[a][b] + k[b][a])));
    let w = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (k[a][b] - k[b][a])));
    (s, w)
}

/// Den snedsymmetriska delen av friktionstangenten (osymmetrisk lösare):
/// K[B₁, B₂] += A c₁ c₂ W.
pub(super) fn add_skew(k: &mut dyn super::TangentSink, coef: &[(usize, f64)], area: f64, w: &[[f64; 3]; 3], dim: usize) {
    for &(b1, c1) in coef {
        for &(b2, c2) in coef {
            let f = area * c1 * c2;
            for p1 in 0..dim {
                for p2 in 0..dim {
                    let v = f * w[p1][p2];
                    if v != 0.0 {
                        k.add_unsym(b1, crate::fem::dof::U1 + p1 as crate::fem::dof::Dof, b2, crate::fem::dof::U1 + p2 as crate::fem::dof::Dof, v);
                    }
                }
            }
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
