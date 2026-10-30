//! Anisotrop linjär elasticitet och lokala materialriktningar.
//!
//! *ELASTIC med TYPE=
//! - ENGINEERING CONSTANTS: E1, E2, E3, ν12, ν13, ν23, G12, G13, G23
//!   (ortotrop, angiven via flexibiliteten),
//! - ORTHOTROPIC: D1111, D1122, D2222, D1133, D2233, D3333, D1212, D1313, D2323,
//! - ANISOTROPIC: de 21 konstanterna i den övre triangeln kolumnvis
//!   (D1111, D1122, D2222, D1133, D2233, D3333, D1112, ..., D2323),
//! - LAMINA: E1, E2, ν12, G12, G13, G23 (bara plan spänning).
//!
//! Konstitutiva sambandet är σ = D ε i komponentordningen (11, 22, 33,
//! 12, 13, 23) med ingenjörsskjuvtöjning, så D1212 = G12 osv. För plan
//! spänning och stänger kondenseras de spänningskomponenter som ska vara noll
//! bort (statisk kondensation av 3D-matrisen), för plan töjning tas
//! delmatrisen för (11, 22, 33, 12).
//!
//! Lokala materialriktningar (*ORIENTATION, SYSTEM=RECTANGULAR eller
//! CYLINDRICAL, DEFINITION=COORDINATES, med valfri extra rotation kring en
//! lokal axel) kopplas till element via tvärsnittets parameter ORIENTATION=.
//! Kontinuumkärnan (`elements/common.rs`) roterar töjningen till de lokala
//! axlarna, anropar materialet och roterar tillbaka spänning och tangent;
//! spänning och töjning redovisas i lokala axlar. Riktningarna
//! beräknas i varje integrationspunkt (spelar roll för cylindriska system).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::elements::common::{cross, dot, inv3, norm, scale, sub};
use crate::fem::model::{rotate_vec, FeModel, MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "ELASTIC (anisotrop)", consumes: &["ELASTIC"], priority: 1, build: Some(build) }];

/// Index för komponenterna i ett spänningstillstånd bland de sex 3D-komponenterna.
pub fn mode_components(mode: StressMode) -> &'static [usize] {
    match mode {
        StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
        StressMode::PlaneStrain | StressMode::Axisymmetric => &[0, 1, 2, 3],
        StressMode::PlaneStress => &[0, 1, 3],
        StressMode::Uniaxial => &[0],
    }
}

/// Komponenter som är spänningsfria (kondenseras bort) i ett spänningstillstånd.
pub fn free_components(mode: StressMode) -> &'static [usize] {
    match mode {
        StressMode::PlaneStress => &[2, 4, 5],
        StressMode::Uniaxial => &[1, 2, 3, 4, 5],
        _ => &[],
    }
}

/// Löser A x = b (n×n radvis) med Gausselimination och pivotering.
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

/// Invers av en n×n-matris (radvis).
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

/// Reducerar en 6×6 styvhetsmatris till spänningstillståndet `mode`:
/// delmatris för givna komponenter, statisk kondensation av spänningsfria.
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

/// Linjärt anisotropt elastiskt material med 6×6-matrisen `d` (lokala axlar),
/// eller en färdig matris för plan spänning (LAMINA).
#[derive(Debug, Clone)]
pub struct AnisotropicElastic {
    pub d: [[f64; 6]; 6],
    /// LAMINA: 3×3-matris för plan spänning (11, 22, 12).
    pub lamina: Option<[f64; 9]>,
    pub kind: String,
}

impl AnisotropicElastic {
    pub fn dmat(&self, mode: StressMode) -> Result<Vec<f64>> {
        if let Some(l) = &self.lamina {
            return match mode {
                StressMode::PlaneStress => Ok(l.to_vec()),
                StressMode::Uniaxial => Ok(vec![l[0] - l[1] * l[3] / l[4]]),
                _ => fail("*ELASTIC, TYPE=LAMINA gäller bara plan spänning (CPS-element och skal)"),
            };
        }
        Ok(reduce_6x6(&self.d, mode))
    }
}

/// Tolkar alla tal i ett *ELASTIC-block (fortsättningsrader slås ihop).
fn all_floats(kw: &RawKeyword) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    for d in &kw.data {
        v.extend(kw.floats(d)?);
    }
    Ok(v)
}

/// 6×6-styvhet för ortotropt material från ingenjörskonstanter.
pub fn engineering_constants(c: &[f64; 9]) -> Result<[[f64; 6]; 6]> {
    let [e1, e2, e3, n12, n13, n23, g12, g13, g23] = *c;
    if !(e1 > 0.0 && e2 > 0.0 && e3 > 0.0 && g12 > 0.0 && g13 > 0.0 && g23 > 0.0) {
        return fail("moduler i ENGINEERING CONSTANTS ska vara positiva");
    }
    let s = [
        [1.0 / e1, -n12 / e1, -n13 / e1],
        [-n12 / e1, 1.0 / e2, -n23 / e2],
        [-n13 / e1, -n23 / e2, 1.0 / e3],
    ];
    let (inv, det) = inv3(s);
    if !(det > 0.0) {
        return fail("ENGINEERING CONSTANTS ger en flexibilitetsmatris som inte är positivt definit (kontrollera ν)");
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
        // TYPE=SHEAR (deviatorisk del till *EOS) hanteras av `materials/eos.rs`
        return Ok(None);
    }
    if ty == "TRACTION" {
        return Ok(None); // kohesiva element: materials/cohesive.rs
    }
    if kw.has("DEPENDENCIES") || kw.has("TEMPERATURE") {
        return kw.err("temperaturberoende elasticitet stöds inte ännu");
    }
    let v = all_floats(kw)?;
    let need = |n: usize| -> Result<()> {
        if v.len() < n {
            return kw.err(format!("TYPE={ty} kräver {n} konstanter, {} angivna", v.len()));
        }
        if v.len() > n + 1 {
            return kw.err(format!("TYPE={ty}: fler än {n} konstanter (temperaturberoende data stöds inte ännu)"));
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
                return kw.err("E1, E2 och G12 ska vara positiva");
            }
            let n21 = n12 * e2 / e1;
            let den = 1.0 - n12 * n21;
            if !(den > 0.0) {
                return kw.err("LAMINA: 1 − ν12 ν21 ska vara positivt");
            }
            lamina = Some([e1 / den, n12 * e2 / den, 0.0, n12 * e2 / den, e2 / den, 0.0, 0.0, 0.0, g12]);
            let mut d = [[0.0; 6]; 6];
            d[3][3] = g12;
            d[4][4] = v[4];
            d[5][5] = v[5];
            d
        }
        other => return kw.err(format!("TYPE={other} stöds inte (ISOTROPIC, ENGINEERING CONSTANTS, ORTHOTROPIC, ANISOTROPIC, LAMINA)")),
    };
    if lamina.is_none() {
        // stabilitet: matrisen ska vara positivt definit
        let flat: Vec<f64> = d.iter().flatten().copied().collect();
        if !positive_definite(&flat, 6) {
            return kw.err(format!("TYPE={ty}: elasticitetsmatrisen är inte positivt definit"));
        }
    }
    Ok(Some(Arc::new(AnisotropicElastic { d, lamina, kind: ty })))
}

/// Cholesky-test av positiv definithet.
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
        "ELASTIC (anisotrop)"
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

/// Ett lokalt koordinatsystem för materialriktningar.
#[derive(Debug, Clone)]
pub struct Orientation {
    pub name: String,
    pub system: OrientationSystem,
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub c: [f64; 3],
    /// Extra rotation (lokal axel 1–3, vinkel i grader).
    pub extra: Option<(usize, f64)>,
}

impl Orientation {
    pub fn parse(kw: &RawKeyword) -> Result<Orientation> {
        let name = kw.param_upper("NAME").unwrap_or_default();
        let system = match kw.param_upper("SYSTEM").as_deref() {
            None | Some("RECTANGULAR") => OrientationSystem::Rectangular,
            Some("CYLINDRICAL") => OrientationSystem::Cylindrical,
            Some(o) => return kw.err(format!("SYSTEM={o} stöds inte ännu (RECTANGULAR, CYLINDRICAL)")),
        };
        match kw.param_upper("DEFINITION").as_deref() {
            None | Some("COORDINATES") => {}
            Some(o) => return kw.err(format!("DEFINITION={o} stöds inte ännu (bara COORDINATES)")),
        }
        if kw.has("LOCAL DIRECTIONS") {
            return kw.err("LOCAL DIRECTIONS stöds inte ännu");
        }
        let recs = kw.records();
        let Some(first) = recs.first() else { return kw.err("datarad med punkterna a och b saknas") };
        let v = kw.floats(first)?;
        if v.len() < 6 {
            return kw.err("datarad 1 ska ha a (3 koordinater) och b (3 koordinater)");
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
                    return kw.err_at(d.line, "den extra rotationens axel ska vara 1, 2 eller 3");
                }
                // axeln behålls även med vinkeln 0 (skal: den anger vilka axlar som
                // ligger i skalets plan, se elements/shell_section.rs)
                Some((axis, ang))
            }
            None => None,
        };
        let o = Orientation { name, system, a, b, c, extra };
        if o.system == OrientationSystem::Rectangular {
            let ax = o.axes_at([0.0; 3]);
            if ax.iter().any(|r| r.iter().any(|x| !x.is_finite())) {
                return kw.err("punkterna a, b och origo ger inget koordinatsystem (parallella eller sammanfallande)");
            }
        }
        Ok(o)
    }

    /// Lokala axlar i punkten `p`: rad i = lokal axel i uttryckt i globala koordinater.
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

    /// Transformationsmatris för töjningar i spänningstillståndet `mode`
    /// (ε_lokal = T ε_global, ingenjörsskjuvtöjning), ncomp × ncomp radvis.
    pub fn strain_transform_mode(&self, p: [f64; 3], mode: StressMode) -> Vec<f64> {
        let t = strain_transform(&self.axes_at(p));
        let idx = mode_components(mode);
        idx.iter().flat_map(|&i| idx.iter().map(move |&j| t[i][j])).collect()
    }
}

/// Komponentpar (i, j) för 3D-ordningen 11, 22, 33, 12, 13, 23.
pub const VOIGT: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

/// 6×6-matrisen T med ε_lokal = T ε_global (ingenjörsskjuvtöjning) för
/// rotationen `q` (rad i = lokal axel i).
pub fn strain_transform(q: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut t = [[0.0; 6]; 6];
    for (jc, &(k, l)) in VOIGT.iter().enumerate() {
        // global töjningstensor för enhetskomponenten jc
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

/// Orientering per tvärsnitt (tvärsnittets ORIENTATION= mot *ORIENTATION i modellen).
pub fn section_orientations(model: &FeModel) -> Result<Vec<Option<Orientation>>> {
    let mut out = Vec::with_capacity(model.sections.len());
    for s in &model.sections {
        let Some(name) = s.raw.param_upper("ORIENTATION") else {
            out.push(None);
            continue;
        };
        let kw = model.extra.iter().find(|k| k.name == "ORIENTATION" && k.param_upper("NAME").as_deref() == Some(name.as_str()));
        let Some(kw) = kw else {
            return s.raw.err(format!("orienteringen '{name}' är inte definierad (*ORIENTATION, NAME={name})"));
        };
        out.push(Some(Orientation::parse(kw)?));
    }
    Ok(out)
}
