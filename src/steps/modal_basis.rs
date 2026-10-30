//! Modbasen för de modala stegen (*MODAL DYNAMIC, *STEADY STATE DYNAMICS,
//! *RESPONSE SPECTRUM): exakt samma egenmoder som det föregående
//! *FREQUENCY-steget, plus modal dämpning, *SELECT EIGENMODES och *BASE MOTION.
//!
//! Moderna räknas om med `frequency::extract` för det närmast föregående
//! *FREQUENCY-steget, med det stegets randvillkor och bastillståndet. Det ger
//! bit för bit samma moder som *FREQUENCY-steget skrev (deterministisk
//! extraktion), med konsistent massa och inre frihetsgrader (B23/B33) där
//! *FREQUENCY har det. Vektorerna normeras till generaliserad massa 1
//! (φᵀMφ = 1 över alla frihetsgrader, även de inre).
//!
//! *SELECT EIGENMODES: DEFINITION=MODE NUMBERS
//! (standard) med en lista av modnummer per datarad, eller med GENERATE
//! "första, sista, steg"; DEFINITION=FREQUENCY RANGE med "lägsta, högsta
//! frekvens" (Hz). Modnumren i *MODAL DAMPING avser de ursprungliga numren;
//! dämpning för moder som inte valts ignoreras.
//!
//! *MODAL DAMPING: VISCOUS (andel av kritisk dämpning, standard), RAYLEIGH
//! (ζ = α/(2ω) + βω/2) och STRUCTURAL (s, imaginär styvhet i ω²(1 + is)).
//! DEFINITION=MODE NUMBERS: "första mod, sista mod, värde" (RAYLEIGH: α, β).
//! DEFINITION=FREQUENCY RANGE: "frekvens (Hz), värde" per rad, linjär
//! interpolation i frekvens mellan raderna och ändvärdet utanför. Olika dämpningstyper adderas; samma typ i flera nyckelord skriver
//! över per mod.
//!
//! *BASE MOTION (huvudbasen): DOF=1–3, AMPLITUDE=, SCALE=, TYPE=ACCELERATION
//! (standard), VELOCITY eller DISPLACEMENT. Rörelseekvationen skrivs relativt
//! basen, u = u_rel + r u_g, så den modala lasten blir −Γᵢ ü_g med
//! deltagandefaktorn Γᵢ = φᵢᵀ M r (r = stelkroppsförskjutning i DOF-riktningen
//! över alla frihetsgrader, även de låsta – samma som PF i *FREQUENCY).
//! U, V, A skrivs relativt basen och TU, TV, TA totalt.

use super::super::frequency::{self, AllDofs};
use crate::fem::assembly::DofMap;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::fem::state::StepLoads;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::SymCsr;
use crate::steps::api::StepCtx;

/// Egenmoder från ett *FREQUENCY-steg (efter *SELECT EIGENMODES).
pub struct ModalBasis {
    pub dm: DofMap,
    pub all: AllDofs,
    /// Massmatrisen över alla frihetsgrader (fria, inre, låsta).
    pub mfull: SymCsr,
    /// Modnummer (1-baserat) i *FREQUENCY-steget.
    pub number: Vec<usize>,
    pub lambda: Vec<f64>,
    pub omega: Vec<f64>,
    /// Modvektorer i `all`-numreringen (längd `all.n()`), φᵀMφ = 1.
    pub x: Vec<Vec<f64>>,
    /// Modvektorernas nodvärden.
    pub phi: Vec<Vec<NodeVec>>,
}

impl ModalBasis {
    /// Moderna från det närmast föregående *FREQUENCY-steget.
    pub fn from_frequency_step(ctx: &StepCtx) -> Result<ModalBasis> {
        let p = ctx.prep;
        let model = p.model;
        let Some(j) = (0..ctx.index).rev().find(|&i| model.steps[i].procedure_name() == "FREQUENCY") else {
            return fail(format!("*{} kräver ett föregående *FREQUENCY-steg", ctx.step.procedure_name()));
        };
        let fs = &model.steps[j];
        let (loads, _) = StepLoads::build_with(model, fs, ctx.state, &p.active, false)?;
        let kw = fs.procedure.as_ref().unwrap();
        let ex = frequency::extract(p, kw, &loads, ctx.state, ctx.state.nlgeom, ctx.solver, false)?;
        let nn = model.nodes.len();
        let selected = select_eigenmodes(ctx.step, &ex.modes.iter().map(|m| m.0).collect::<Vec<_>>())?;
        let mut b = ModalBasis { dm: ex.dm, all: ex.all, mfull: ex.mfull, number: Vec::new(), lambda: Vec::new(), omega: Vec::new(), x: Vec::new(), phi: Vec::new() };
        let ntot = b.all.n();
        for (i, (lam, v)) in ex.modes.into_iter().enumerate() {
            if !selected[i] {
                continue;
            }
            let mut x = v;
            x.resize(ntot, 0.0);
            let gm = b.mfull.bilinear(&x, &x);
            if gm > 0.0 {
                let f = 1.0 / gm.sqrt();
                x.iter_mut().for_each(|y| *y *= f);
            }
            b.phi.push(b.all.to_nodes(&x, nn));
            b.x.push(x);
            b.number.push(i + 1);
            b.lambda.push(lam);
            b.omega.push(lam.max(0.0).sqrt());
        }
        if b.x.is_empty() {
            return fail("inga egenmoder valda (*FREQUENCY/*SELECT EIGENMODES)");
        }
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Modala laster φᵢᵀ F för nodkrafter F.
    pub fn modal_force(&self, f: &[NodeVec]) -> Vec<f64> {
        self.phi.iter().map(|phi| phi.iter().zip(f).map(|(a, b)| (0..SLOTS).map(|s| a[s] * b[s]).sum::<f64>()).sum()).collect()
    }

    /// Σ φᵢ qᵢ som nodvektor.
    pub fn superpose(&self, q: &[f64]) -> Vec<NodeVec> {
        let nn = self.phi.first().map_or(0, |p| p.len());
        let mut u = vec![[0.0; SLOTS]; nn];
        for (phi, &qi) in self.phi.iter().zip(q) {
            if qi == 0.0 {
                continue;
            }
            for (un, pn) in u.iter_mut().zip(phi) {
                for s in 0..SLOTS {
                    un[s] += pn[s] * qi;
                }
            }
        }
        u
    }

    /// Deltagandefaktorer Γᵢ = φᵢᵀ M r för stelkroppsförskjutningen r i
    /// riktningen `dir` (0–2 translation, 3–5 rotation kring origo).
    pub fn participation(&self, nodes: &[[f64; 3]], dir: usize) -> Vec<f64> {
        let r = frequency::rigid(&self.all, nodes, dir);
        let mr = self.mfull.mul(&r);
        self.x.iter().map(|x| x.iter().zip(&mr).map(|(a, b)| a * b).sum()).collect()
    }

    /// M r (alla frihetsgrader) för stelkroppsförskjutningen i riktningen `dir`.
    pub fn mass_times_rigid(&self, nodes: &[[f64; 3]], dir: usize) -> Vec<f64> {
        self.mfull.mul(&frequency::rigid(&self.all, nodes, dir))
    }
}

/// Valda moder (index i *FREQUENCY-stegets modlista) enligt *SELECT EIGENMODES.
fn select_eigenmodes(step: &StepDef, lambda: &[f64]) -> Result<Vec<bool>> {
    let kws: Vec<&RawKeyword> = step.extra.iter().filter(|k| k.name == "SELECT EIGENMODES").collect();
    if kws.is_empty() {
        return Ok(vec![true; lambda.len()]);
    }
    let mut sel = vec![false; lambda.len()];
    let two_pi = 2.0 * std::f64::consts::PI;
    for kw in kws {
        let def = kw.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
        for d in &kw.data {
            let v = kw.floats(d)?;
            match def.as_str() {
                "MODE NUMBERS" => {
                    let list: Vec<usize> = if kw.has("GENERATE") {
                        if v.len() < 2 {
                            return kw.err_at(d.line, "GENERATE: skriv 'första mod, sista mod[, steg]'");
                        }
                        let inc = v.get(2).copied().filter(|&x| x >= 1.0).unwrap_or(1.0).round() as usize;
                        (v[0].round() as usize..=v[1].round() as usize).step_by(inc).collect()
                    } else {
                        v.iter().map(|x| x.round() as usize).collect()
                    };
                    for m in list {
                        if m >= 1 && m <= sel.len() {
                            sel[m - 1] = true;
                        }
                    }
                }
                "FREQUENCY RANGE" => {
                    if v.len() < 2 {
                        return kw.err_at(d.line, "skriv 'lägsta frekvens, högsta frekvens'");
                    }
                    for (i, l) in lambda.iter().enumerate() {
                        let f = l.max(0.0).sqrt() / two_pi;
                        if f >= v[0] && f <= v[1] {
                            sel[i] = true;
                        }
                    }
                }
                other => return kw.err(format!("DEFINITION={other} ska vara MODE NUMBERS eller FREQUENCY RANGE")),
            }
        }
    }
    Ok(sel)
}

/// Modal dämpning per vald mod.
#[derive(Debug, Clone)]
pub struct Damping {
    /// Andel av kritisk viskös dämpning (VISCOUS + RAYLEIGH).
    pub zeta: Vec<f64>,
    /// Strukturell dämpning s (ω²(1 + i s)).
    pub structural: Vec<f64>,
}

/// Linjär interpolation i en tabell (x, y) sorterad i x, ändvärden utanför.
fn interp(t: &[(f64, f64)], x: f64) -> f64 {
    if x <= t[0].0 {
        return t[0].1;
    }
    for w in t.windows(2) {
        if x <= w[1].0 {
            let d = w[1].0 - w[0].0;
            return if d > 0.0 { w[0].1 + (w[1].1 - w[0].1) * (x - w[0].0) / d } else { w[1].1 };
        }
    }
    t[t.len() - 1].1
}

/// Dämpning från stegets *MODAL DAMPING för modbasen `b`.
pub fn modal_damping(step: &StepDef, b: &ModalBasis) -> Result<Damping> {
    let n = b.len();
    // samma DEFINITION krävs i *SELECT EIGENMODES och *MODAL DAMPING i samma steg
    let def_of = |k: &RawKeyword| k.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
    if let Some(sel) = step.extra.iter().find(|k| k.name == "SELECT EIGENMODES") {
        if let Some(md) = step.extra.iter().find(|k| k.name == "MODAL DAMPING" && def_of(k) != def_of(sel)) {
            return md.err("DEFINITION= måste vara samma i *SELECT EIGENMODES och *MODAL DAMPING i samma steg");
        }
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    // per typ: viskös, Rayleigh (som ζ), strukturell
    let mut vals = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for kw in step.extra.iter().filter(|k| k.name == "MODAL DAMPING") {
        if kw.has("COMPOSITE") || kw.param_upper("MODAL").is_some_and(|m| m != "DIRECT") {
            return kw.err("MODAL=COMPOSITE stöds inte (bara MODAL=DIRECT)");
        }
        if let Some(v) = kw.param_upper("VISCOUS") {
            if v != "FRACTION OF CRITICAL DAMPING" {
                return kw.err(format!("VISCOUS={v} stöds inte (FRACTION OF CRITICAL DAMPING)"));
            }
        }
        let kind = if kw.has("STRUCTURAL") {
            2
        } else if kw.has("RAYLEIGH") {
            1
        } else {
            0
        };
        let def = kw.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
        match def.as_str() {
            "MODE NUMBERS" => {
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    if v.len() < 3 {
                        return kw.err_at(d.line, "skriv 'första mod, sista mod, värde'");
                    }
                    let a = v[0].round() as usize;
                    let last = if v[1] > 0.0 { v[1].round() as usize } else { a };
                    for i in 0..n {
                        let m = b.number[i];
                        if m < a || m > last {
                            continue;
                        }
                        vals[kind][i] = if kind == 1 {
                            let w = b.omega[i];
                            if w > 0.0 { v[2] / (2.0 * w) + v.get(3).copied().unwrap_or(0.0) * w / 2.0 } else { 0.0 }
                        } else {
                            v[2]
                        };
                    }
                }
            }
            "FREQUENCY RANGE" => {
                if kind == 1 {
                    return kw.err("RAYLEIGH med DEFINITION=FREQUENCY RANGE stöds inte ännu");
                }
                let mut t: Vec<(f64, f64)> = Vec::new();
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    if v.len() < 2 {
                        return kw.err_at(d.line, "skriv 'frekvens, värde'");
                    }
                    t.push((v[0], v[1]));
                }
                if t.is_empty() {
                    return kw.err("datarader saknas");
                }
                t.sort_by(|a, b| a.0.total_cmp(&b.0));
                for i in 0..n {
                    vals[kind][i] = interp(&t, b.omega[i] / two_pi);
                }
            }
            other => return kw.err(format!("DEFINITION={other} ska vara MODE NUMBERS eller FREQUENCY RANGE")),
        }
    }
    let zeta = (0..n).map(|i| vals[0][i] + vals[1][i]).collect();
    Ok(Damping { zeta, structural: vals[2].clone() })
}

/// Typ av basrörelse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseKind {
    Acceleration,
    Velocity,
    Displacement,
}

/// *BASE MOTION (huvudbasen).
#[derive(Debug, Clone)]
pub struct BaseMotion {
    /// Riktning 0–2.
    pub dir: usize,
    pub amplitude: Option<String>,
    pub scale: f64,
    pub kind: BaseKind,
}

/// Stegets *BASE MOTION.
pub fn base_motions(step: &StepDef) -> Result<Vec<BaseMotion>> {
    let mut out = Vec::new();
    for kw in step.extra.iter().filter(|k| k.name == "BASE MOTION") {
        if kw.has("BASE NAME") {
            return kw.err("sekundära baser (BASE NAME=) stöds inte");
        }
        let dof = kw.param("DOF").and_then(parse_f64).ok_or_else(|| FemError(format!("{}: DOF= saknas", kw.location())))?.round() as usize;
        if !(1..=3).contains(&dof) {
            return kw.err("bara translationer (DOF=1, 2 eller 3) stöds");
        }
        let scale = match kw.param("SCALE") {
            Some(s) => parse_f64(s).ok_or_else(|| FemError(format!("{}: SCALE={s} är inget tal", kw.location())))?,
            None => 1.0,
        };
        let kind = match kw.param_upper("TYPE").as_deref().unwrap_or("ACCELERATION") {
            "ACCELERATION" => BaseKind::Acceleration,
            "VELOCITY" => BaseKind::Velocity,
            "DISPLACEMENT" => BaseKind::Displacement,
            t => return kw.err(format!("TYPE={t} ska vara ACCELERATION, VELOCITY eller DISPLACEMENT")),
        };
        if kw.has("LOADING") && kw.param_upper("LOADING").as_deref() != Some("UNIFORM") {
            return kw.err("bara LOADING=UNIFORM stöds");
        }
        // basrörelsen kräver en amplitud
        let Some(amplitude) = kw.param_upper("AMPLITUDE") else {
            return kw.err("AMPLITUDE= saknas (basrörelsen kräver en amplitud)");
        };
        out.push(BaseMotion { dir: dof - 1, amplitude: Some(amplitude), scale, kind });
    }
    Ok(out)
}

/// Lägger till r·v (riktning `dir`) på alla noders translation i `dir`
/// (för TU, TV, TA).
pub fn add_rigid(u: &mut [NodeVec], dir: usize, v: f64) {
    for x in u.iter_mut() {
        x[dir] += v;
    }
}
