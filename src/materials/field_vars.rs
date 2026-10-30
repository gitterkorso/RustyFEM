//! Fördefinierade fältvariabler (field variables):
//!
//! - `*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=n`: nodvärden vid start
//!   (noder som inte anges har värdet 0),
//! - `*FIELD, VARIABLE=n` i ett steg: nodvärden vid stegets slut
//!   (OP=MOD standard, OP=NEW återställer variabeln till startvärdena).
//!   Värdena rampas linjärt över steget från föregående värde (eller sätts
//!   direkt med AMPLITUDE=STEP på *STEP) och ärvs till följande steg,
//! - `*FIELD, USER, VARIABLE=n`: värdena i noderna ges av UFIELD
//!   (`user::routines::Ufield`) vid varje tidpunkt.
//!
//! Nodreferenserna löses upp när filen läses (`inp/keywords/user2.rs`):
//! dataraderna blir [nodindex, värde] med parametern `_NODINDEX` och
//! variabelnumret i `VARIABLE`. Materialen läser fälten i punkten via
//! `user::context` (elementets nodvärden, interpolerade i
//! integrationspunkten) och interpolerar materialdata med DEPENDENCIES=
//! (`field_dependent.rs`).

use crate::fem::model::{FeModel, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

/// Datarader med upplösta noder (samma markör som temperaturfältet).
pub const RESOLVED: &str = crate::materials::thermal::RESOLVED;

pub static FACTORIES: &[super::MaterialFactory] = &[];

#[derive(Clone)]
struct UserPart {
    /// Variabel (0-baserad), noder och namnet som rutinen väljs med.
    var: usize,
    nodes: Vec<usize>,
    routine: Arc<dyn crate::user::routines::Ufield>,
}

#[derive(Clone)]
struct Segment {
    start: f64,
    period: f64,
    f0: Vec<f64>,
    f1: Vec<f64>,
    ramp: bool,
    kstep: i32,
    user: Vec<UserPart>,
}

/// Fältvariablerna i noderna som funktion av total tid (nn × nfield, nodvis).
#[derive(Clone)]
pub struct FieldHistory {
    nf: usize,
    initial: Vec<f64>,
    segments: Vec<Segment>,
    labels: Vec<i64>,
    coords: Vec<[f64; 3]>,
}

fn variable(kw: &RawKeyword) -> Result<usize> {
    match kw.param("VARIABLE") {
        None => Ok(1),
        Some(v) => match crate::inp::parse_label(v) {
            Some(n) if n >= 1 => Ok(n as usize),
            _ => kw.err(format!("VARIABLE={v} ska vara ett heltal ≥ 1")),
        },
    }
}

fn apply(kw: &RawKeyword, nf: usize, v: &mut [f64]) -> Result<()> {
    let k = variable(kw)? - 1;
    for d in &kw.data {
        let x = kw.floats(d)?;
        let n = x[0] as usize;
        if (n + 1) * nf <= v.len() {
            v[n * nf + k] = x.get(1).copied().unwrap_or(0.0);
        }
    }
    Ok(())
}

fn timeless(p: &str) -> bool {
    matches!(p, "FREQUENCY" | "BUCKLE" | "COMPLEX FREQUENCY" | "MODAL DYNAMIC" | "STEADY STATE DYNAMICS" | "RESPONSE SPECTRUM")
}

impl FieldHistory {
    /// Fälthistorien, eller `None` om modellen saknar fältvariabler.
    pub fn build(model: &FeModel) -> Result<Option<FieldHistory>> {
        let is_ic = |k: &&RawKeyword| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED) && k.param_upper("TYPE").as_deref() == Some("FIELD");
        let ic: Vec<&RawKeyword> = model.extra.iter().filter(is_ic).collect();
        let fields: Vec<&RawKeyword> = model.steps.iter().flat_map(|s| s.extra.iter().filter(|k| k.name == "FIELD")).collect();
        if ic.is_empty() && fields.is_empty() {
            return Ok(None);
        }
        let mut nf = 0;
        for k in ic.iter().chain(fields.iter()) {
            nf = nf.max(variable(k)?);
        }
        let nn = model.nodes.len();
        let mut initial = vec![0.0; nn * nf];
        for k in &ic {
            apply(k, nf, &mut initial)?;
        }
        let mut cur = initial.clone();
        let mut time = 0.0;
        let mut segments = Vec::new();
        for (i, s) in model.steps.iter().enumerate() {
            if s.perturbation || timeless(s.procedure_name()) {
                continue;
            }
            let period = crate::user::context::step_period(s);
            let mut next = cur.clone();
            let mut user = Vec::new();
            for k in s.extra.iter().filter(|k| k.name == "FIELD") {
                let var = variable(k)? - 1;
                if k.param_upper("OP").as_deref() == Some("NEW") {
                    for n in 0..nn {
                        next[n * nf + var] = initial[n * nf + var];
                    }
                }
                if k.has("USER") {
                    let name = k.param("_NAME").unwrap_or("").to_string();
                    let Some(routine) = crate::user::routines::find_ufield(&name)? else {
                        return crate::user::routines::missing("UFIELD", "nodsetet", &name);
                    };
                    let nodes = k.data.iter().filter_map(|d| k.floats(d).ok().and_then(|x| x.first().map(|&n| n as usize))).collect();
                    user.push(UserPart { var, nodes, routine });
                    continue;
                }
                apply(k, nf, &mut next)?;
            }
            let ramp = !s.amplitude_step && !super::time_dependence::default_step_amplitude(s);
            segments.push(Segment { start: time, period, f0: cur.clone(), f1: next.clone(), ramp, kstep: i as i32 + 1, user });
            time += period;
            cur = next;
        }
        Ok(Some(FieldHistory { nf, initial, segments, labels: model.node_label.clone(), coords: model.nodes.clone() }))
    }

    /// Antal fältvariabler (största VARIABLE=).
    pub fn nfield(&self) -> usize {
        self.nf
    }

    /// Fältvariablerna i noderna `nodes` vid total tid `time` (nodvis, `nfield` per nod).
    pub fn at(&self, nodes: &[usize], time: f64) -> Vec<f64> {
        let nf = self.nf;
        let mut out = Vec::with_capacity(nodes.len() * nf);
        let seg = self.segments.iter().find(|s| time <= s.start + s.period * (1.0 + 1e-12));
        let seg = match seg {
            _ if time <= 0.0 || self.segments.is_empty() => {
                for &n in nodes {
                    out.extend_from_slice(&self.initial[n * nf..(n + 1) * nf]);
                }
                // UFIELD gäller även vid stegets början
                if let Some(s) = self.segments.first() {
                    self.apply_user(s, nodes, time.max(0.0), &mut out);
                }
                return out;
            }
            Some(s) => s,
            None => {
                let s = self.segments.last().unwrap();
                for &n in nodes {
                    out.extend_from_slice(&s.f1[n * nf..(n + 1) * nf]);
                }
                self.apply_user(s, nodes, time, &mut out);
                return out;
            }
        };
        let f = ((time - seg.start) / seg.period).clamp(0.0, 1.0);
        let a = if seg.ramp { f } else if f > 0.0 { 1.0 } else { 0.0 };
        for &n in nodes {
            for k in 0..nf {
                let (x0, x1) = (seg.f0[n * nf + k], seg.f1[n * nf + k]);
                out.push(x0 + a * (x1 - x0));
            }
        }
        self.apply_user(seg, nodes, time, &mut out);
        out
    }

    /// UFIELD för noderna i stegets *FIELD, USER.
    fn apply_user(&self, seg: &Segment, nodes: &[usize], time: f64, out: &mut [f64]) {
        if seg.user.is_empty() {
            return;
        }
        let nf = self.nf;
        let kinc = crate::user::context::step_inc().1;
        for (i, &n) in nodes.iter().enumerate() {
            for u in seg.user.iter().filter(|u| u.nodes.contains(&n)) {
                let field = &mut out[i * nf..(i + 1) * nf];
                let mut a = crate::user::routines::UfieldArgs {
                    field,
                    kfield: u.var + 1,
                    kstep: seg.kstep,
                    kinc,
                    time: [time - seg.start, time],
                    node: self.labels.get(n).copied().unwrap_or(0),
                    coords: self.coords.get(n).copied().unwrap_or([0.0; 3]),
                    temp: 0.0,
                    dtemp: 0.0,
                    nfield: nf,
                };
                // fel i rutinen kan inte returneras härifrån: värdet behålls
                if let Err(e) = u.routine.ufield(&mut a) {
                    eprintln!("varning: UFIELD i nod {}: {e}", a.node);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation i temperatur och fältvariabler (DEPENDENCIES=)
// ---------------------------------------------------------------------------

/// Tabell med beroenden: varje post har egenskapsvärden följda av de
/// oberoende variablerna (inre först: t.ex. [T, f1, ..., fn], för *PLASTIC
/// [ε̄p, T, f1, ..., fn]). Posterna står i ordning: den inre variabeln
/// varierar snabbast, den yttre (fn) långsammast.
#[derive(Debug, Clone)]
pub struct DepTable {
    pub nprops: usize,
    pub nvars: usize,
    pub rows: Vec<Vec<f64>>,
}

impl DepTable {
    /// Läser posterna ur dataraderna: `nprops` egenskaper, temperatur och
    /// `ndep` fältvariabler per post; högst 8 värden per rad (indataregel,
    /// längre poster fortsätter på nästa rad). `inner` extra oberoende
    /// variabler står mellan egenskaperna och temperaturen (*PLASTIC: ε̄p).
    pub fn parse(kw: &RawKeyword, nprops: usize, inner: usize, ndep: usize) -> Result<DepTable> {
        let per = nprops + inner + 1 + ndep;
        let lines_per = per.div_ceil(8);
        let mut rows = Vec::new();
        let data = &kw.data;
        let mut i = 0;
        while i < data.len() {
            let mut v = Vec::new();
            for k in 0..lines_per {
                let Some(d) = data.get(i + k) else {
                    return kw.err_at(data[i].line, format!("posten ska ha {per} värden (DEPENDENCIES={ndep})"));
                };
                let mut x = kw.floats(d)?;
                x.truncate(8);
                if k + 1 < lines_per {
                    x.resize(8, 0.0);
                }
                v.extend(x);
            }
            v.resize(per, 0.0);
            rows.push(v);
            i += lines_per;
        }
        if rows.is_empty() {
            return kw.err("datarad saknas");
        }
        Ok(DepTable { nprops: nprops + inner, nvars: 1 + ndep, rows })
    }

    /// Värdena (de `nprops` första kolumnerna) vid de oberoende variablerna `x`
    /// (samma ordning som i posten). Hierarkisk linjär interpolation: först i
    /// den yttersta variabeln mellan grupperna med samma värde, sedan inåt;
    /// konstant utanför tabellen.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let idx: Vec<usize> = (0..self.rows.len()).collect();
        eval_rows(&self.rows, &idx, self.nprops, self.nvars, x)
    }
}

/// Interpolerar posterna `idx` i variabel `level` (0 = inre) och inåt.
fn eval_rows(rows: &[Vec<f64>], idx: &[usize], np: usize, level_count: usize, x: &[f64]) -> Vec<f64> {
    if level_count == 0 || idx.len() == 1 {
        return rows[idx[0]][..np].to_vec();
    }
    let lv = level_count - 1;
    let col = np + lv;
    // grupper med samma värde i variabeln (i ordning)
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for &r in idx {
        let v = rows[r][col];
        match groups.last_mut() {
            Some((g, members)) if *g == v => members.push(r),
            _ => groups.push((v, vec![r])),
        }
    }
    let xv = x.get(lv).copied().unwrap_or(0.0);
    if groups.len() == 1 || xv <= groups[0].0 {
        return eval_rows(rows, &groups[0].1, np, lv, x);
    }
    let last = groups.len() - 1;
    if xv >= groups[last].0 {
        return eval_rows(rows, &groups[last].1, np, lv, x);
    }
    let k = groups.windows(2).position(|w| xv >= w[0].0 && xv <= w[1].0).unwrap_or(0);
    let (a, b) = (&groups[k], &groups[k + 1]);
    let s = if b.0 > a.0 { (xv - a.0) / (b.0 - a.0) } else { 0.0 };
    let va = eval_rows(rows, &a.1, np, lv, x);
    let vb = eval_rows(rows, &b.1, np, lv, x);
    va.iter().zip(&vb).map(|(p, q)| p + s * (q - p)).collect()
}

/// Felmeddelande om fälthistorien används utan fält.
pub fn require_fields() -> Result<()> {
    fail("materialet har DEPENDENCIES= men modellen har inga fältvariabler (*FIELD, *INITIAL CONDITIONS, TYPE=FIELD)")
}
