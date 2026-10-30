//! Begynnelsevillkor i integrationspunkterna: *INITIAL CONDITIONS,
//! TYPE=STRESS (även GEOSTATIC) och TYPE=RATIO (portal).
//!
//! Nyckelorden läses av `inp/keywords/geomaterials.rs`, som löser upp
//! element- och nodreferenserna och sparar dem i `FeModel::extra` med
//! parametern `RESOLVED`:
//! - TYPE=STRESS: rader [elementindex, σ1, ..., σn] (komponentordningen
//!   för elementtypen), lika i alla integrationspunkter;
//! - TYPE=STRESS, GEOSTATIC: rader [elementindex, σv1, y1, σv2, y2, k1, k2]:
//!   vertikal spänning linjär i den vertikala koordinaten (y i 2D, z i 3D),
//!   horisontella spänningar k·σv (k2 = k1 om den saknas);
//! - TYPE=RATIO: rader [nodindex, e1, y1, e2, y2] (portal linjärt i nodens
//!   vertikala koordinat, konstant om bara e1 anges), interpolerat till
//!   integrationspunkterna med formfunktionerna.
//!
//! `apply` anropas av `steps::run` efter `Prepared::init_states`: spänningen
//! skrivs i elementtillståndets spänningsdel och materialet får
//! `MaterialModel::init_state_at`. GEOSTATIC-spänningen och portalet tas i
//! elementets mittpunkt (nodmedelvärdet) och är lika i alla dess
//! integrationspunkter.

use super::{MaterialFactory, StressMode};
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::fem::model::FeModel;
use crate::fem::state::SolutionState;
use crate::model::{fail, FemError, Result};

/// Modulen har inga egna materialfabriker.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Parameter som markerar ett upplöst *INITIAL CONDITIONS (STRESS/RATIO).
pub const RESOLVED: &str = "RUSTYFEM_GEO";

/// Nyckel i `SolutionState::vectors` för begynnelsespänningens inre krafter
/// F0 = ∫Bᵀσ0 dV. F0·(1 − t) läggs till de yttre krafterna under
/// första steget (med AMPLITUDE=STEP på lasterna blir svaret
/// σ = F_ext + F0(1 − t), med rampade laster som balanserar σ0 ingen
/// deformation), så att en begynnelsespänning som inte är i jämvikt avvecklas
/// linjärt; utdata i frame 0 har inte korrektionen. Tas bort efter första
/// steget (`steps::run`).
pub const UNBALANCE: &str = "INITIAL_STRESS_UNBALANCE";

/// F_ext += F0 (1 − t).
pub fn add_unbalance(fext: &mut [crate::fem::dof::NodeVec], f0: &[crate::fem::dof::NodeVec], t: f64) {
    for (a, b) in fext.iter_mut().zip(f0) {
        for s in 0..a.len() {
            a[s] += b[s] * (1.0 - t);
        }
    }
}

/// Upplöst spänningsdefinition för ett element.
#[derive(Clone)]
enum StressDef {
    Uniform(Vec<f64>),
    /// σv1, y1, σv2, y2, k1, k2
    Geostatic([f64; 6]),
    /// *INITIAL CONDITIONS, TYPE=STRESS, USER: SIGINI (`user::routines`), elementsetets namn.
    User(String),
}

/// Finns det begynnelsevillkor av typ STRESS eller RATIO i modellen?
pub fn present(model: &FeModel) -> bool {
    model.extra.iter().any(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED))
}

/// Nodernas portal (NaN = inte givet).
fn nodal_void_ratio(model: &FeModel, vertical: impl Fn(usize) -> usize) -> Result<Vec<f64>> {
    let mut e = vec![f64::NAN; model.nodes.len()];
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED) && k.param_upper("TYPE").as_deref() == Some("RATIO")) {
        for d in &kw.data {
            let v = kw.floats(d)?;
            let n = v[0] as usize;
            let val = if v.len() >= 5 && v[4] != v[2] {
                let y = model.nodes[n][vertical(n)];
                v[1] + (v[3] - v[1]) * (y - v[2]) / (v[4] - v[2])
            } else {
                v[1]
            };
            e[n] = val;
        }
    }
    Ok(e)
}

/// Sätter begynnelsespänning och portal i elementens tillstånd och sparar
/// begynnelsespänningens inre krafter (`UNBALANCE`).
pub fn apply(prep: &Prepared, state: &mut SolutionState) -> Result<()> {
    let model = prep.model;
    if !present(model) {
        return Ok(());
    }
    let any_stress = set_states(prep, &mut state.elem_state)?;
    if any_stress {
        let mut f0 = vec![[0.0; crate::fem::dof::SLOTS]; model.nodes.len()];
        for e in 0..model.elements.len() {
            let ctx = prep.ctx(e, false, 0.0, 0.0);
            if ctx.material.is_none() || state.elem_state[e].is_empty() {
                continue;
            }
            let et = model.elements[e].etype;
            let z = vec![0.0; et.n_dofs()];
            let mut sn = Vec::new();
            let r = et.internal_force(&ctx, &z, &z, &state.elem_state[e], &mut sn, false).map_err(|err| FemError(format!("{}: {}", model.elem_name(e), err.0)))?;
            for (k, &(n, s)) in prep.edofs[e].iter().enumerate() {
                f0[n][s] += r.fint[k];
            }
        }
        state.vectors.insert(UNBALANCE.to_string(), f0);
    }
    Ok(())
}

/// Skriver begynnelsevillkoren i elementtillstånden. Returnerar om någon
/// begynnelsespänning är satt.
fn set_states(prep: &Prepared, states: &mut [Vec<f64>]) -> Result<bool> {
    let model = prep.model;
    let mut any_stress = false;
    let ne = model.elements.len();
    // vertikal riktning per element: y i 2D/axisymmetri, z i 3D
    let vdir = |e: usize| -> usize {
        match model.elements[e].etype.family() {
            Family::Continuum3D | Family::Shell | Family::Membrane => 2,
            _ => 1,
        }
    };
    let mut node_dir = vec![2usize; model.nodes.len()];
    for (e, el) in model.elements.iter().enumerate() {
        for &n in &el.nodes {
            node_dir[n] = vdir(e);
        }
    }
    let void = nodal_void_ratio(model, |n| node_dir[n])?;
    let mut sdef: Vec<Option<StressDef>> = vec![None; ne];
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED) && k.param_upper("TYPE").as_deref() == Some("STRESS")) {
        let geo = kw.has("GEOSTATIC");
        let user = kw.has("USER");
        for d in &kw.data {
            let v = kw.floats(d)?;
            let e = v[0] as usize;
            sdef[e] = Some(if user {
                StressDef::User(kw.param("_NAME").unwrap_or("").to_string())
            } else if geo {
                let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
                let k1 = g(5);
                let k2 = if v.len() > 6 { g(6) } else { k1 };
                StressDef::Geostatic([g(1), g(2), g(3), g(4), k1, k2])
            } else {
                StressDef::Uniform(v[1..].to_vec())
            });
        }
    }
    for e in 0..ne {
        let has_void = model.elements[e].nodes.iter().any(|&n| !void[n].is_nan());
        if sdef[e].is_none() && !has_void {
            continue;
        }
        // porvattentryckselementen sköter sina begynnelsevillkor själva (steps/soils.rs)
        if crate::elements::soils::lookup(model.elements[e].etype.name()).is_some() {
            continue;
        }
        let ctx = prep.ctx(e, false, 0.0, 0.0);
        let Some(mat) = ctx.material else { continue };
        let et = model.elements[e].etype;
        let nip = et.n_ips();
        let len = states[e].len();
        let ns = mat.n_state();
        let fam = et.family();
        let nc = [6usize, 4, 3, 1].into_iter().find(|&nc| nip > 0 && len == nip * (nc + ns));
        let Some(nc) = nc else {
            return fail(format!(
                "{}: *INITIAL CONDITIONS, TYPE=STRESS/RATIO stöds inte för elementtypen {}",
                model.elem_name(e),
                et.name()
            ));
        };
        let mode = match (nc, fam) {
            (6, _) => StressMode::ThreeD,
            (4, Family::Axisymmetric) => StressMode::Axisymmetric,
            (4, _) => StressMode::PlaneStrain,
            (3, _) => StressMode::PlaneStress,
            _ => StressMode::Uniaxial,
        };
        let nodes = &model.elements[e].nodes;
        // elementets medelvärde i alla integrationspunkter (för GEOSTATIC
        // och portalet: lika värden i elementets punkter)
        let at_ip = |_ip: usize, f: &dyn Fn(usize) -> f64| -> f64 { nodes.iter().map(|&n| f(n)).sum::<f64>() / nodes.len() as f64 };
        let vd = vdir(e);
        let stride = nc + ns;
        // SIGINI: integrationspunkternas lägen (elementet rapporterar dem till user::context)
        let ip_coords = match &sdef[e] {
            Some(StressDef::User(_)) => {
                let z = vec![0.0; et.n_dofs()];
                let st0 = states[e].clone();
                crate::user::context::record_ips(|| {
                    let mut sn = Vec::new();
                    let _ = et.internal_force(&ctx, &z, &z, &st0, &mut sn, false);
                })
            }
            _ => Vec::new(),
        };
        for ip in 0..nip {
            let mut s = vec![0.0; nc];
            match &sdef[e] {
                Some(StressDef::Uniform(v)) => {
                    if v.len() > nc && v[nc..].iter().any(|&x| x != 0.0) {
                        return fail(format!("{}: för många spänningskomponenter i *INITIAL CONDITIONS, TYPE=STRESS för {}", model.elem_name(e), et.name()));
                    }
                    for (i, &x) in v.iter().take(nc).enumerate() {
                        s[i] = x;
                    }
                }
                Some(StressDef::Geostatic([s1, y1, s2, y2, k1, k2])) => {
                    let y = at_ip(ip, &|n| model.nodes[n][vd]);
                    let sv = if y2 != y1 { s1 + (s2 - s1) * (y - y1) / (y2 - y1) } else { *s1 };
                    match mode {
                        StressMode::ThreeD => {
                            s[0] = k1 * sv;
                            s[1] = k2 * sv;
                            s[2] = sv;
                        }
                        StressMode::PlaneStress => {
                            s[0] = k1 * sv;
                            s[1] = sv;
                        }
                        StressMode::Uniaxial => s[0] = sv,
                        _ => {
                            s[0] = k1 * sv;
                            s[1] = sv;
                            s[2] = k2 * sv;
                        }
                    }
                }
                Some(StressDef::User(name)) => {
                    let Some(routine) = crate::user::routines::find_sigini(name)? else {
                        return crate::user::routines::missing("SIGINI", "elementsetet", name);
                    };
                    let coords = ip_coords.iter().find(|(k, _, _)| *k == ip + 1).map(|c| c.1).unwrap_or_else(|| crate::elements::common::centroid(&ctx.x));
                    let ndim = match fam {
                        Family::Continuum3D | Family::Shell | Family::Membrane => 3,
                        _ => 2,
                    };
                    let mut a = crate::user::routines::SiginiArgs {
                        sigma: &mut s,
                        coords,
                        ntens: nc,
                        ncrds: ndim,
                        noel: model.elements[e].label,
                        npt: ip + 1,
                        layer: 1,
                        kspt: 1,
                        lrebar: 0,
                        names: ["", name],
                    };
                    routine.sigini(&mut a).map_err(|err| FemError(format!("{}: SIGINI: {}", model.elem_name(e), err.0)))?;
                }
                None => {}
            }
            let ev = if has_void {
                if nodes.iter().any(|&n| void[n].is_nan()) {
                    return fail(format!("{}: portalet (*INITIAL CONDITIONS, TYPE=RATIO) saknas i några av elementets noder", model.elem_name(e)));
                }
                Some(at_ip(ip, &|n| void[n]))
            } else {
                None
            };
            any_stress |= s.iter().any(|&x| x != 0.0);
            let base = ip * stride;
            let st = &mut states[e];
            st[base..base + nc].copy_from_slice(&s);
            mat.init_state_at(mode, &s, ev, &mut st[base + nc..base + stride])
                .map_err(|err| FemError(format!("{}: {}", model.elem_name(e), err.0)))?;
        }
    }
    Ok(any_stress)
}
