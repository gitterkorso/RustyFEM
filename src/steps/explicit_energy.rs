//! Energibalans och history-utdata (*OUTPUT, HISTORY + *ENERGY OUTPUT) för
//! explicit dynamik, med standardnamnen och regionen "Assembly ASSEMBLY":
//!
//! - ALLKE = ½ u̇ᵀ M u̇ med hastigheten vid hela inkrementet.
//! - ALLIE = Σ ½ (Iᵢ + Iᵢ₊₁)·Δu: elementens (och fjädrarnas) inre arbete,
//!   ALLSE + ALLPD + ALLAE.
//! - ALLAE: hourglass-energin för reducerat integrerade solider (0 för
//!   övriga element; skalens hourglass-energi ingår i ALLSE).
//! - ALLVD = Σ ½ (Cᵢ + Cᵢ₊₁)·Δu: viskös dissipation (volymviskositet,
//!   Rayleigh-dämpning, dämpare).
//! - ALLFD: friktionsdissipation i kontakt; ALLPW: arbetet av kontaktens
//!   straffkrafter (negativt när energi lagras i kontakten).
//! - ALLWK = Σ ½ (Pᵢ + Pᵢ₊₁)·Δu + reaktionernas arbete i föreskrivna
//!   frihetsgrader.
//! - ETOTAL = ALLKE + ALLIE + ALLVD + ALLFD − ALLWK − ALLPW.
//!
//! ALLSE är ALLIE − ALLAE när alla material är linjära. Med olinjära material
//! (plasticitet) i kontinuumelement är ALLSE den elastiska töjningsenergin
//! Σ ½ σ : C⁻¹ : σ · V/n över integrationspunkterna (aktuell volym vid NLGEOM,
//! lika vikter) och ALLPD = ALLIE − ALLSE − ALLAE (`elastic_energy`); för andra
//! element med olinjära material skrivs varken ALLSE eller ALLPD. History skrivs vid t = 0 och sedan i första inkrementet
//! vid eller efter varje utdatatid (TIME INTERVAL, NUMBER INTERVAL, standard
//! 200 intervall) eller var n:te inkrement (FREQUENCY).

use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, StepDef};
use crate::model::Result;

pub const REGION: &str = "Assembly ASSEMBLY";

/// Ackumulerade energier.
#[derive(Debug, Clone, Copy, Default)]
pub struct Energies {
    pub ke: f64,
    pub ie: f64,
    pub ae: f64,
    pub vd: f64,
    pub fd: f64,
    pub pw: f64,
    pub wk: f64,
    /// Elastisk töjningsenergi (olinjära material), annars `None`.
    pub se: Option<f64>,
}

impl Energies {
    pub fn etotal(&self) -> f64 {
        self.ke + self.ie + self.vd + self.fd - self.wk - self.pw
    }
}

/// ½ (a + b)·d summerat över noder och slots 0–6.
pub fn work(a: &[NodeVec], b: &[NodeVec], d: &[NodeVec]) -> f64 {
    let mut s = 0.0;
    for n in 0..d.len() {
        for k in 0..7 {
            s += 0.5 * (a[n][k] + b[n][k]) * d[n][k];
        }
    }
    s
}

#[derive(Debug, Clone, Copy)]
enum When {
    Interval(f64),
    Every(usize),
}

/// Begärd history-utdata för hela modellen.
pub struct History {
    when: When,
    vars: Vec<String>,
    next: f64,
    pub data: Vec<(String, Vec<(f64, f64)>)>,
}

/// Energinamn som RustyFEM kan skriva.
const ALL: &[&str] = &["ALLAE", "ALLFD", "ALLIE", "ALLKE", "ALLPD", "ALLPW", "ALLSE", "ALLVD", "ALLWK", "ETOTAL"];

impl History {
    /// Läser *OUTPUT, HISTORY med *ENERGY OUTPUT (utan ELSET) ur steget.
    pub fn from_step(step: &StepDef, period: f64, linear: bool, se_ok: bool) -> Result<Option<History>> {
        let mut when: Option<When> = None;
        let mut vars: Vec<String> = Vec::new();
        let mut any = false;
        for kw in &step.output {
            if kw.name == "OUTPUT" {
                when = None;
                if kw.has("HISTORY") {
                    let num = |k: &str| kw.param(k).and_then(parse_f64);
                    when = Some(if let Some(n) = num("FREQUENCY") {
                        When::Every((n.round() as usize).max(1))
                    } else if let Some(n) = num("NUMBER INTERVAL") {
                        When::Interval(period / n.max(1.0))
                    } else if let Some(d) = num("TIME INTERVAL") {
                        When::Interval(d)
                    } else {
                        When::Interval(period / 200.0)
                    });
                    if kw.param_upper("VARIABLE").is_some_and(|v| v == "PRESELECT" || v == "ALL") {
                        any = true;
                        vars.extend(ALL.iter().map(|s| s.to_string()));
                    }
                }
            } else if kw.name == "ENERGY OUTPUT" && when.is_some() && !kw.has("ELSET") {
                any = true;
                let mut list: Vec<String> = kw.data.iter().flat_map(|d| d.fields.iter()).map(|f| f.trim().to_uppercase()).filter(|f| !f.is_empty()).collect();
                if list.is_empty() || kw.param_upper("VARIABLE").is_some_and(|v| v == "PRESELECT" || v == "ALL") {
                    list = ALL.iter().map(|s| s.to_string()).collect();
                }
                vars.extend(list);
            }
        }
        if !any {
            return Ok(None);
        }
        vars.sort();
        vars.dedup();
        // ALLSE/ALLPD bara när de kan beräknas
        vars.retain(|v| ALL.contains(&v.as_str()) && (linear || se_ok || v != "ALLSE") && (!linear && se_ok || v != "ALLPD"));
        let when = when.unwrap_or(When::Interval(period / 200.0));
        let next = match when {
            When::Interval(d) => d,
            When::Every(_) => f64::INFINITY,
        };
        let data = vars.iter().map(|v| (v.clone(), Vec::new())).collect();
        Ok(Some(History { when, vars, next, data }))
    }

    fn record(&mut self, t: f64, e: &Energies) {
        for (i, v) in self.vars.iter().enumerate() {
            let val = match v.as_str() {
                "ALLKE" => e.ke,
                "ALLIE" => e.ie,
                "ALLAE" => e.ae,
                "ALLSE" => e.se.unwrap_or(e.ie - e.ae),
                "ALLPD" => e.ie - e.ae - e.se.unwrap_or(e.ie - e.ae),
                "ALLVD" => e.vd,
                "ALLFD" => e.fd,
                "ALLPW" => e.pw,
                "ALLWK" => e.wk,
                "ETOTAL" => e.etotal(),
                _ => continue,
            };
            self.data[i].1.push((t, val));
        }
    }

    /// Behövs den elastiska töjningsenergin (ALLSE/ALLPD med olinjära material)?
    pub fn wants_se(&self) -> bool {
        self.vars.iter().any(|v| v == "ALLSE" || v == "ALLPD")
    }

    /// Anropas vid t = 0 och efter varje inkrement; `last` = stegets sista inkrement.
    pub fn after_increment(&mut self, t: f64, inc: usize, last: bool, eps: f64, e: &Energies) {
        if inc == 0 {
            self.record(t, e);
            return;
        }
        let write = match self.when {
            When::Every(k) => inc % k == 0,
            When::Interval(d) => {
                let hit = t >= self.next - eps;
                if hit {
                    while self.next <= t + eps {
                        self.next += d;
                    }
                }
                hit
            }
        };
        if write || last {
            self.record(t, e);
        }
    }

}

/// Elastisk töjningsenergi i kontinuumelement med isotrop elasticitet (se
/// modulens dokumentation). `None` om något element med olinjärt material inte
/// är ett kontinuumelement med isotrop elasticitet.
pub fn elastic_energy(p: &crate::fem::assembly::Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, state: &[Vec<f64>]) -> Option<f64> {
    use crate::elements::Family;
    use crate::materials::StressMode;
    let m = p.model;
    let mut w = 0.0;
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let Some(mat) = ctx.material else { continue };
        let fam = el.etype.family();
        let name = el.etype.name();
        let mode = match fam {
            Family::Continuum3D => StressMode::ThreeD,
            Family::Continuum2D if name.starts_with("CPS") => StressMode::PlaneStress,
            Family::Continuum2D if name.starts_with("CPE") => StressMode::PlaneStrain,
            _ if mat.is_linear() => return None,
            _ => return None,
        };
        let (ey, nu) = mat.elastic_moduli()?;
        let nips = el.etype.n_ips();
        let stride = crate::elements::common::ip_stride(mode, mat);
        let st = state.get(e)?;
        if nips == 0 || st.len() < nips * stride {
            continue;
        }
        let g = super::dt::geometry(fam, name, &x(e))?;
        let t = if fam == Family::Continuum2D { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 };
        let dv = g.vol * t / nips as f64;
        let nc = mode.ncomp();
        for q in 0..nips {
            let sg = &st[q * stride..q * stride + nc];
            let (nn, sh): (&[f64], &[f64]) = match mode {
                StressMode::PlaneStress => (&sg[..2], &sg[2..]),
                _ => (&sg[..3], &sg[3..]),
            };
            let tr: f64 = nn.iter().sum();
            let ss: f64 = nn.iter().map(|v| v * v).sum::<f64>() + 2.0 * sh.iter().map(|v| v * v).sum::<f64>();
            w += 0.5 / ey * ((1.0 + nu) * ss - nu * tr * tr) * dv;
        }
    }
    Some(w)
}
