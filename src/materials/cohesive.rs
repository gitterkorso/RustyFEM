//! Traction–separation för kohesiva element: *ELASTIC,
//! TYPE=TRACTION med *DAMAGE INITIATION (CRITERION=MAXS, QUADS, MAXE, QUADE)
//! och *DAMAGE EVOLUTION (TYPE=DISPLACEMENT eller ENERGY).
//!
//! Lagen arbetar i elementets lokala riktningar (n, s, t) med nominella
//! töjningar ε = δ/T0 (δ = separation, T0 = konstitutiv tjocklek från
//! *COHESIVE SECTION) och dragspänningar t:
//!
//! - Elasticitet (okopplad): t̄n = Knn εn, t̄s = Kss εs, t̄t = Ktt εt.
//! - Initiering (⟨x⟩ = max(x, 0), ingen initiering i tryck):
//!   MAXS: max(⟨t̄n⟩/tn0, t̄s/ts0, t̄t/tt0) = 1, QUADS: Σ(t̄ᵢ/tᵢ0)² = 1,
//!   MAXE/QUADE samma med töjningarna. Utdata MAXSCRT/QUADSCRT/... är
//!   kriteriets största värde hittills och 1 efter initieringen.
//! - Effektiv separation δm = √(⟨δn⟩² + δs² + δt²). Initieringspunkten
//!   räknas längs aktuell separationsriktning (kriteriet är homogent i
//!   skalan: δm0 = λ δm med λ = 1/f respektive 1/√f) och den effektiva
//!   dragspänningen vid initieringen som arbetskonjugat
//!   Teff0 = t̄0·δ0/δm0 (kontrollerat med blandad mod).
//! - Utveckling med δmax = största δm hittills:
//!   TYPE=DISPLACEMENT anger δf − δm0 (förskjutning, inte töjning),
//!   TYPE=ENERGY anger brottenergin Gc (δf = 2Gc/Teff0 vid linjär mjukning);
//!   MIXED MODE BEHAVIOR=BK (Gc = Gn + (Gs − Gn)(GS/GT)^η, POWER=η) eller
//!   POWER LAW (Σ(Gᵢ/Gᵢc)^α = 1, POWER=α) med energiandelarna
//!   Gᵢ ∝ ⟨t̄ᵢ⟩δᵢ. SOFTENING=LINEAR: D = δf(δmax − δm0)/(δmax(δf − δm0)),
//!   EXPONENTIAL (förskjutning, α): D = 1 − δm0/δmax·[1 − (1 − e^(−α(δmax−δm0)/(δf−δm0)))/(1 − e^(−α))],
//!   EXPONENTIAL (energi): D = ∫Teff dδ/(Gc − G0), G0 = ½Teff0 δm0
//!   (trapetsregel över inkrementen).
//! - D växer aldrig. Viskös regularisering (*SECTION CONTROLS, VISCOSITY=μ):
//!   Dv = (μ Dv,n + Δt D)/(μ + Δt) används i spänningen; formeln för D
//!   används även bortom δf och Dv begränsas därefter till 1.
//! - tn = (1 − D) t̄n i drag, t̄n i tryck; ts = (1 − D) t̄s, tt = (1 − D) t̄t.
//!   D = 1 betyder att punkten har gått sönder; elementet tas bort (STATUS = 0)
//!   när alla punkter gått sönder och ingen är i tryck (ett element
//!   i tryck finns kvar och bär trycket).
//!
//! Materialet används bara av de kohesiva elementen (`elements/cohesive.rs`)
//! via `MaterialModel::cohesive`; i andra element ger det fel.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{parse_f64, MaterialDef, RawKeyword};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "traction-separation (kohesiva element)",
    consumes: &["ELASTIC", "DAMAGE INITIATION", "DAMAGE EVOLUTION"],
    priority: 30,
    build: Some(build),
}];

/// Initieringskriterier för kohesiva element.
pub const CRITERIA: &[&str] = &["MAXS", "QUADS", "MAXE", "QUADE"];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Crit {
    MaxS,
    QuadS,
    MaxE,
    QuadE,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mix {
    /// Oberoende av moden.
    None,
    Bk { eta: f64 },
    Power { alpha: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Softening {
    Linear,
    /// Exponentiell med α (TYPE=DISPLACEMENT).
    Exp(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Evol {
    /// δf − δm0.
    Disp { du: f64, soft: Softening },
    /// Gn, Gs, Gt (lika om modberoende saknas).
    Energy { g: [f64; 3], mix: Mix, exp: bool },
}

#[derive(Debug, Clone)]
pub struct TractionSeparation {
    k: [f64; 3],
    init: Option<(Crit, [f64; 3])>,
    evol: Option<Evol>,
}

// Tillstånd per integrationspunkt.
const D: usize = 0; // skada utan viskositet (växer aldrig)
const DV: usize = 1; // skada som används (viskös)
const DMAX: usize = 2; // största effektiva separation
const CRIT: usize = 3; // kriteriets värde (utdata)
const BROKEN: usize = 4; // 1 när punkten gått sönder (D = 1)
const W: usize = 5; // ∫Teff dδ (exponentiell energimjukning)
const NSTATE: usize = 6;

/// Resultat av en uppdatering: dragspänning och tangent dt/dε (3×3).
#[derive(Debug, Clone, Copy)]
pub struct CohOut {
    pub t: [f64; 3],
    pub dtde: [[f64; 3]; 3],
}

fn values(kw: &RawKeyword) -> Result<Vec<f64>> {
    if kw.has("DEPENDENCIES") || kw.data.len() > 1 {
        return kw.err("temperatur-, fält- eller modberoende data stöds inte ännu för kohesiva element");
    }
    kw.first_floats()
}

fn param_f64(kw: &RawKeyword, key: &str) -> Result<Option<f64>> {
    match kw.param(key) {
        None => Ok(None),
        Some(s) => parse_f64(s).map(Some).ok_or_else(|| FemError(format!("{}: ogiltigt {key}=", kw.location()))),
    }
}

/// Sant om materialet är en traction–separationslag (TYPE=TRACTION eller ett kohesivt kriterium).
pub fn is_cohesive(def: &MaterialDef) -> bool {
    def.behavior("ELASTIC").is_some_and(|k| k.param_upper("TYPE").as_deref() == Some("TRACTION"))
        || def
            .behaviors
            .iter()
            .any(|b| b.name == "DAMAGE INITIATION" && b.param_upper("CRITERION").is_some_and(|c| CRITERIA.contains(&c.as_str())))
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if super::xfem_damage::claims(def) {
        return Ok(None); // XFEM-spricka: materials/xfem_damage.rs
    }
    if !is_cohesive(def) {
        return Ok(None);
    }
    let Some(ek) = def.behavior("ELASTIC").filter(|k| k.param_upper("TYPE").as_deref() == Some("TRACTION")) else {
        return fail(format!("material {}: kohesiva skadekriterier kräver *ELASTIC, TYPE=TRACTION", def.name));
    };
    let v = values(ek)?;
    let kn = v.first().copied().unwrap_or(0.0);
    let ks = v.get(1).copied().filter(|&x| x != 0.0).unwrap_or(kn);
    let kt = v.get(2).copied().filter(|&x| x != 0.0).unwrap_or(ks);
    if !(kn > 0.0 && ks > 0.0 && kt > 0.0) {
        return ek.err("styvheterna Knn, Kss, Ktt ska vara positiva");
    }
    let mut init = None;
    let mut evol = None;
    for b in &def.behaviors {
        match b.name.as_str() {
            "DAMAGE INITIATION" => {
                if init.is_some() {
                    return b.err("bara ett initieringskriterium stöds för kohesiva element");
                }
                let c = match b.param_upper("CRITERION").as_deref() {
                    Some("MAXS") => Crit::MaxS,
                    Some("QUADS") => Crit::QuadS,
                    Some("MAXE") => Crit::MaxE,
                    Some("QUADE") => Crit::QuadE,
                    Some(o) => return b.err(format!("CRITERION={o} stöds inte för kohesiva element (MAXS, QUADS, MAXE, QUADE)")),
                    None => return b.err("CRITERION= saknas"),
                };
                let v = values(b)?;
                let a = v.first().copied().unwrap_or(0.0);
                let s = v.get(1).copied().filter(|&x| x != 0.0).unwrap_or(a);
                let t = v.get(2).copied().filter(|&x| x != 0.0).unwrap_or(s);
                if !(a > 0.0 && s > 0.0 && t > 0.0) {
                    return b.err("initieringsvärdena ska vara positiva");
                }
                init = Some((c, [a, s, t]));
            }
            "DAMAGE EVOLUTION" => {
                if init.is_none() {
                    return b.err("*DAMAGE EVOLUTION kräver en föregående *DAMAGE INITIATION");
                }
                if evol.is_some() {
                    return b.err("kriteriet har redan en *DAMAGE EVOLUTION");
                }
                evol = Some(evolution(b)?);
            }
            _ => {}
        }
    }
    Ok(Some(Arc::new(TractionSeparation { k: [kn, ks, kt], init, evol })))
}

fn evolution(b: &RawKeyword) -> Result<Evol> {
    let ty = b.param_upper("TYPE").unwrap_or_else(|| "DISPLACEMENT".into());
    let soft = b.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
    if let Some(d) = b.param_upper("DEGRADATION").filter(|d| d != "MAXIMUM") {
        return b.err(format!("DEGRADATION={d} stöds inte för kohesiva element"));
    }
    if b.param_upper("MODE MIX RATIO").is_some_and(|m| m != "ENERGY") {
        return b.err("MODE MIX RATIO=TRACTION stöds inte ännu (bara ENERGY)");
    }
    let v = values(b)?;
    match ty.as_str() {
        "DISPLACEMENT" => {
            if b.has("MIXED MODE BEHAVIOR") {
                return b.err("MIXED MODE BEHAVIOR med TYPE=DISPLACEMENT (modberoende tabell) stöds inte ännu");
            }
            let du = v.first().copied().unwrap_or(0.0);
            if !(du > 0.0) {
                return b.err("brottförskjutningen ska vara positiv");
            }
            let soft = match soft.as_str() {
                "LINEAR" => Softening::Linear,
                "EXPONENTIAL" => {
                    let a = v.get(1).copied().unwrap_or(0.0);
                    if !(a > 0.0) {
                        return b.err("SOFTENING=EXPONENTIAL kräver exponenten α > 0");
                    }
                    Softening::Exp(a)
                }
                s => return b.err(format!("SOFTENING={s} stöds inte ännu (LINEAR, EXPONENTIAL)")),
            };
            Ok(Evol::Disp { du, soft })
        }
        "ENERGY" => {
            let exp = match soft.as_str() {
                "LINEAR" => false,
                "EXPONENTIAL" => true,
                s => return b.err(format!("SOFTENING={s} stöds inte ännu (LINEAR, EXPONENTIAL)")),
            };
            let mix = match b.param_upper("MIXED MODE BEHAVIOR").as_deref() {
                None => Mix::None,
                Some("BK") => Mix::Bk { eta: param_f64(b, "POWER")?.ok_or_else(|| FemError(format!("{}: MIXED MODE BEHAVIOR=BK kräver POWER=", b.location())))? },
                Some("POWER LAW") => Mix::Power { alpha: param_f64(b, "POWER")?.ok_or_else(|| FemError(format!("{}: MIXED MODE BEHAVIOR=POWER LAW kräver POWER=", b.location())))? },
                Some(o) => return b.err(format!("MIXED MODE BEHAVIOR={o} stöds inte (BK, POWER LAW)")),
            };
            let gn = v.first().copied().unwrap_or(0.0);
            let g = if mix == Mix::None {
                [gn; 3]
            } else {
                let gs = v.get(1).copied().filter(|&x| x != 0.0).unwrap_or(gn);
                [gn, gs, v.get(2).copied().filter(|&x| x != 0.0).unwrap_or(gs)]
            };
            if g.iter().any(|&x| !(x > 0.0)) {
                return b.err("brottenergierna ska vara positiva");
            }
            Ok(Evol::Energy { g, mix, exp })
        }
        t => b.err(format!("TYPE={t} stöds inte (DISPLACEMENT, ENERGY)")),
    }
}

impl TractionSeparation {
    pub fn n_state(&self) -> usize {
        NSTATE
    }

    pub fn init_state(&self, st: &mut [f64]) {
        for v in st.iter_mut() {
            *v = 0.0;
        }
    }

    pub fn has_damage(&self) -> bool {
        self.evol.is_some()
    }

    /// Utdatanamnet för initieringskriteriet.
    pub fn criterion_name(&self) -> Option<&'static str> {
        self.init.map(|(c, _)| match c {
            Crit::MaxS => "MAXSCRT",
            Crit::QuadS => "QUADSCRT",
            Crit::MaxE => "MAXECRT",
            Crit::QuadE => "QUADECRT",
        })
    }

    /// (SDEG, kriterium, sönder) ur tillståndet.
    pub fn outputs(&self, st: &[f64]) -> (f64, f64, bool) {
        if st.len() < NSTATE {
            return (0.0, 0.0, false);
        }
        (st[DV], st[CRIT], st[BROKEN] > 0.5)
    }

    /// Punkten tillhör ett borttaget element.
    pub fn is_deleted(&self, st: &[f64]) -> bool {
        st.len() >= NSTATE && st[BROKEN] > 1.5
    }

    /// Markerar att elementet har tagits bort (punkten ger då ingen spänning alls).
    pub fn mark_deleted(&self, st: &mut [f64]) {
        st[BROKEN] = 2.0;
    }

    /// Elastisk tangent dt/dε (diagonal).
    pub fn elastic(&self) -> [[f64; 3]; 3] {
        let mut d = [[0.0; 3]; 3];
        for i in 0..3 {
            d[i][i] = self.k[i];
        }
        d
    }

    /// Kriteriets värde f för odämpade dragspänningen `tb` och töjningen `e`
    /// (nd komponenter) och gradens homogenitet (1 eller 2).
    fn criterion(&self, tb: &[f64; 3], e: &[f64; 3], nd: usize) -> Option<(f64, f64)> {
        let (c, lim) = self.init?;
        let val = |i: usize| {
            let x = match c {
                Crit::MaxS | Crit::QuadS => tb[i],
                Crit::MaxE | Crit::QuadE => e[i],
            };
            let x = if i == 0 { x.max(0.0) } else { x.abs() };
            // 2D: komponenterna (n, s); t saknas
            if i >= nd { 0.0 } else { x / lim[i] }
        };
        Some(match c {
            Crit::MaxS | Crit::MaxE => ((0..3).map(val).fold(0.0, f64::max), 1.0),
            Crit::QuadS | Crit::QuadE => ((0..3).map(|i| val(i) * val(i)).sum(), 2.0),
        })
    }

    /// Spänningsuppdatering. `e` = nominella töjningar (n, s, t), `nd` = 2 eller 3,
    /// `t0` = konstitutiv tjocklek (separation δ = ε T0), `dt` = tidsinkrement,
    /// `visc` = viskositet μ (0 = ingen).
    pub fn update(&self, e: &[f64; 3], nd: usize, t0: f64, dt: f64, visc: f64, old: &[f64], new: &mut [f64]) -> CohOut {
        let (mut out, growing) = self.eval(e, nd, t0, dt, visc, old, new);
        if growing {
            // skadan beror på separationens riktning (δm0, δf och modblandningen):
            // numerisk tangent med framåtdifferenser
            let scale = e.iter().take(nd).fold(0.0f64, |m, x| m.max(x.abs()));
            let h = 1e-7 * scale;
            let mut tmp = vec![0.0; NSTATE];
            for j in 0..nd {
                let mut ep = *e;
                ep[j] += h;
                let (o2, _) = self.eval(&ep, nd, t0, dt, visc, old, &mut tmp);
                for i in 0..nd {
                    out.dtde[i][j] = (o2.t[i] - out.t[i]) / h;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn eval(&self, e: &[f64; 3], nd: usize, t0: f64, dt: f64, visc: f64, old: &[f64], new: &mut [f64]) -> (CohOut, bool) {
        new[..NSTATE].copy_from_slice(&old[..NSTATE]);
        let mut e = *e;
        for x in e.iter_mut().skip(nd) {
            *x = 0.0;
        }
        let k = self.k;
        let tb = [k[0] * e[0], k[1] * e[1], k[2] * e[2]];
        if old[BROKEN] > 0.5 {
            // punkten är sönder: bara tryck i normalriktningen (inget alls när elementet är borttaget)
            let mut out = CohOut { t: [0.0; 3], dtde: [[0.0; 3]; 3] };
            if e[0] < 0.0 && old[BROKEN] < 1.5 {
                out.t[0] = tb[0];
                out.dtde[0][0] = k[0];
            }
            return (out, false);
        }
        let Some((f, deg)) = self.criterion(&tb, &e, nd) else {
            return (CohOut { t: tb, dtde: self.elastic() }, false);
        };
        let initiated = old[CRIT] >= 1.0 || f >= 1.0;
        let Some(evol) = self.evol else {
            new[CRIT] = old[CRIT].max(f);
            return (CohOut { t: tb, dtde: self.elastic() }, false);
        };
        // största värdet hittills, 1 efter initieringen
        new[CRIT] = if initiated { 1.0 } else { old[CRIT].max(f) };
        // separationer och effektiv separation
        let dl = [e[0].max(0.0) * t0, e[1] * t0, e[2] * t0];
        let dm = (dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]).sqrt();
        let mut dnew = old[D];
        let mut growing = false;
        if initiated && dm > 0.0 && f > 0.0 {
            let dmax_old = old[DMAX];
            let dmax = dmax_old.max(dm);
            new[DMAX] = dmax;
            let lam = if deg == 1.0 { 1.0 / f } else { 1.0 / f.sqrt() };
            let dm0 = lam * dm;
            // arbetskonjugerad effektiv dragspänning vid initieringen
            let work = tb[0].max(0.0) * dl[0] + tb[1] * dl[1] + tb[2] * dl[2];
            let teff0 = lam * work / dm;
            let dfun = |x: f64, wold: f64| -> (f64, f64) {
                // (D, ∫Teff dδ) vid δmax = x
                if x <= dm0 {
                    return (0.0, wold);
                }
                match evol {
                    Evol::Disp { du, soft } => {
                        let df = dm0 + du;
                        (lin_or_exp(soft, x, dm0, df), wold)
                    }
                    Evol::Energy { g, mix, exp } => {
                        let gc = critical_energy(g, mix, &tb, &dl);
                        if !exp {
                            let df = (2.0 * gc / teff0).max(dm0 * (1.0 + 1e-12));
                            (lin_or_exp(Softening::Linear, x, dm0, df), wold)
                        } else {
                            // D = ∫Teff dδ/(Gc − G0), trapets från förra δmax (Teff = (1 − D) Keff δ)
                            let g0 = 0.5 * teff0 * dm0;
                            let keff = teff0 / dm0;
                            let (a, ta) = if dmax_old > dm0 { (dmax_old, (1.0 - old[D]) * keff * dmax_old) } else { (dm0, teff0) };
                            let den = gc - g0;
                            if !(den > 0.0) {
                                return (1.0, wold);
                            }
                            // D (Gc − G0) = W + ½(ta + (1 − D) keff x)(x − a)
                            let h = 0.5 * (x - a);
                            let dval = (wold + h * (ta + keff * x)) / (den + h * keff * x);
                            let dval = dval.clamp(0.0, 1.0);
                            (dval, wold + h * (ta + (1.0 - dval) * keff * x))
                        }
                    }
                }
            };
            let wold = old[W];
            let (dcand, wnew) = dfun(dmax, wold);
            if dcand > old[D] {
                dnew = dcand;
                new[W] = wnew;
                growing = dm >= dmax_old && (dnew < 1.0 || visc > 0.0);
            }
        }
        new[D] = dnew;
        let dv = if visc > 0.0 { (visc * old[DV] + dt.max(0.0) * dnew) / (visc + dt.max(0.0)) } else { dnew };
        let dv = dv.min(1.0);
        new[DV] = dv;
        if dv >= 1.0 {
            new[BROKEN] = 1.0;
        }
        let mut out = CohOut { t: [0.0; 3], dtde: [[0.0; 3]; 3] };
        for i in 0..nd {
            let damaged = i > 0 || e[0] >= 0.0;
            let s = if damaged { 1.0 - dv } else { 1.0 };
            out.t[i] = s * tb[i];
            out.dtde[i][i] = s * k[i];
        }
        (out, growing)
    }
}

/// Skadan vid δmax = x. Formeln används även bortom δf (D > 1: D begränsas
/// till 1 först efter den viskösa regulariseringen).
fn lin_or_exp(soft: Softening, x: f64, d0: f64, df: f64) -> f64 {
    match soft {
        Softening::Linear => (df * (x - d0) / (x * (df - d0))).max(0.0),
        Softening::Exp(a) => {
            let r = (x - d0) / (df - d0);
            (1.0 - d0 / x * (1.0 - (1.0 - (-a * r).exp()) / (1.0 - (-a).exp()))).max(0.0)
        }
    }
}

/// Brottenergin för aktuell modblandning (energiandelar Gᵢ ∝ ⟨t̄ᵢ⟩δᵢ).
fn critical_energy(g: [f64; 3], mix: Mix, tb: &[f64; 3], dl: &[f64; 3]) -> f64 {
    let gi = [tb[0].max(0.0) * dl[0], tb[1] * dl[1], tb[2] * dl[2]];
    let gt: f64 = gi.iter().sum();
    if !(gt > 0.0) {
        return g[0];
    }
    match mix {
        Mix::None => g[0],
        Mix::Bk { eta } => {
            let ms = (gi[1] + gi[2]) / gt;
            // BK: Gc = Gn + (Gs − Gn)(GS/GT)^η; skjuvenergin viktas mellan Gs och Gt
            let gsc = if gi[1] + gi[2] > 0.0 { (gi[1] * g[1] + gi[2] * g[2]) / (gi[1] + gi[2]) } else { g[1] };
            g[0] + (gsc - g[0]) * ms.powf(eta)
        }
        Mix::Power { alpha } => {
            let s: f64 = (0..3).map(|i| (gi[i] / gt / g[i]).powf(alpha)).sum();
            s.powf(-1.0 / alpha)
        }
    }
}

impl MaterialModel for TractionSeparation {
    fn name(&self) -> &str {
        "ELASTIC, TYPE=TRACTION"
    }

    fn is_linear(&self) -> bool {
        self.evol.is_none()
    }

    fn requires_unsymmetric(&self) -> bool {
        // blandad mod ger en osymmetrisk tangent efter initieringen
        self.evol.is_some()
    }

    fn update(&self, _input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        fail("*ELASTIC, TYPE=TRACTION kan bara användas i kohesiva element (COH2D4, COH3D8, COH3D6, COHAX4) med *COHESIVE SECTION, RESPONSE=TRACTION SEPARATION")
    }

    fn tangent_linear(&self, _mode: StressMode) -> Result<Vec<f64>> {
        fail("*ELASTIC, TYPE=TRACTION kan bara användas i kohesiva element med *COHESIVE SECTION, RESPONSE=TRACTION SEPARATION")
    }

    fn cohesive(&self) -> Option<&TractionSeparation> {
        Some(self)
    }
}
