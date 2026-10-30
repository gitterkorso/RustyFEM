//! Brygga från `.rusty`-modellen (`crate::model::Model`) till kärnan.
//!
//! En `.rusty`-modell översätts till en `FeModel` med ett allmänt *STATIC-steg:
//! randvillkoren som *BOUNDARY och alla laster (inklusive tryck och egenvikt)
//! som ekvivalenta nodlaster (*CLOAD), precis som vid exporten till .inp.
//! Sedan löses den med step-ramverket och resultatet görs om till `Results`.

use crate::elements::{self, beam_eb, ElemCtx};
use crate::fem::assembly::Prepared;
use crate::fem::dof::Dof;
use crate::fem::model::{BcBlock, BeamSectionDef, CloadBlock, FeModel, MaterialDef, RawKeyword, SectionDef, SectionKind, StepDef, FLAT_INSTANCE};
use crate::materials::elastic::IsotropicElastic;
use crate::model::{fail, Dim, FemError, Model, Result, Section};
use crate::results::Results;
use crate::steps::{self, RunOptions};

/// Elementtypens namn i .inp-formatet (CPS → CPE vid plan töjning).
pub fn elem_name(m: &Model, ei: usize) -> String {
    m.elem_kind(&m.elements[ei]).type_name()
}

fn fe_section(m: &Model, si: usize) -> SectionDef {
    let s = &m.sections[si];
    let kind = match s {
        Section::Solid { thickness, .. } => SectionKind::Solid { thickness: (m.dim == Dim::Two).then_some(*thickness) },
        Section::Truss { area, .. } => SectionKind::Solid { thickness: Some(*area) },
        Section::Beam { profile, n1, .. } => SectionKind::Beam(BeamSectionDef {
            shape: "GENERAL".into(),
            profile: profile.clone(),
            i12: 0.0,
            n1: Some(*n1),
            moduli: None,
            density: 0.0,
        }),
    };
    SectionDef {
        elset: format!("E{}", si + 1),
        material: format!("M{}", s.material() + 1),
        material_index: Some(s.material()),
        kind,
        raw: RawKeyword { name: "SECTION".into(), file: "(.rusty)".into(), ..Default::default() },
    }
}

/// ∫Nᵢ dV per nod för element `ei` (för egenvikt i `Model::nodal_loads`).
pub fn volume_weights(m: &Model, ei: usize) -> Result<Vec<f64>> {
    let e = &m.elements[ei];
    let name = elem_name(m, ei);
    let et = elements::lookup(&name).ok_or_else(|| FemError(format!("intern: elementtypen {name} saknas")))?;
    let sec = fe_section(m, e.section);
    let mat = &m.materials[m.sections[e.section].material()];
    let iso = IsotropicElastic { e: mat.e, nu: mat.nu };
    let ctx = ElemCtx {
        index: ei,
        label: ei as i64 + 1,
        x: e.nodes.iter().map(|&n| m.nodes[n]).collect(),
        section: Some(&sec),
        material: Some(&iso),
        density: mat.density,
        nlgeom: false,
        time: 0.0,
        dt: 0.0,
        temp: Vec::new(),
        temp_old: Vec::new(),
        orientation: None,
    };
    et.volume_weights(&ctx)
}

/// Översätter modellen till en `FeModel` (utan steg).
pub fn to_fe(m: &Model) -> Result<FeModel> {
    let mut fe = FeModel::new();
    let inst = fe.instance(FLAT_INSTANCE);
    for (i, &p) in m.nodes.iter().enumerate() {
        fe.add_node(inst, i as i64 + 1, p);
    }
    for (i, mt) in m.materials.iter().enumerate() {
        if !(mt.e > 0.0) {
            return fail(format!("materialet '{}' saknar E", mt.name));
        }
        fe.materials.push(MaterialDef::elastic(&format!("M{}", i + 1), mt.e, mt.nu, mt.density));
    }
    for si in 0..m.sections.len() {
        fe.sections.push(fe_section(m, si));
    }
    let mut sets: Vec<Vec<usize>> = vec![Vec::new(); m.sections.len()];
    for (i, e) in m.elements.iter().enumerate() {
        let name = elem_name(m, i);
        let et = elements::lookup(&name).ok_or_else(|| FemError(format!("intern: elementtypen {name} saknas")))?;
        sets[e.section].push(fe.add_element(et, inst, i as i64 + 1, e.nodes.clone())?);
    }
    for (si, set) in sets.iter().enumerate() {
        fe.add_to_elset(&format!("E{}", si + 1), set);
    }
    fe.finalize()?;
    Ok(fe)
}

/// Översätter modellen till en `FeModel` med ett linjärt statiskt steg
/// (stöd och laster från skriptet).
pub fn to_fe_static(m: &Model) -> Result<FeModel> {
    m.check()?;
    let active = m.active_dofs();
    let cons = m.constraints(&active)?;
    let fext = m.nodal_loads(&active)?;
    let mut fe = to_fe(m)?;
    let mut step = StepDef {
        name: "STATIC".into(),
        procedure: Some(RawKeyword { name: "STATIC".into(), file: "(.rusty)".into(), ..Default::default() }),
        max_increments: 1,
        ..Default::default()
    };
    step.bcs.push(BcBlock { items: cons.iter().map(|&(n, d, v)| (n, d as Dof + 1, v)).collect(), source: "(.rusty) stöd".into(), ..Default::default() });
    let mut cl = CloadBlock { source: "(.rusty) laster".into(), ..Default::default() };
    for (n, f) in fext.iter().enumerate() {
        for d in 0..6 {
            if f[d] != 0.0 && active[n] & (1 << d) != 0 {
                cl.items.push((n, d as Dof + 1, f[d]));
            }
        }
    }
    step.cloads.push(cl);
    fe.steps.push(step);
    Ok(fe)
}

/// Löser en `.rusty`-modell (linjär statik).
pub fn solve(m: &Model) -> Result<Results> {
    let active = m.active_dofs();
    let cons = m.constraints(&active)?;
    let fext = m.nodal_loads(&active)?;
    let t0 = std::time::Instant::now();
    let fe = to_fe_static(m)?;
    if crate::settings::with(|s| s.timing) {
        eprintln!("till FeModel: {:.3} s", t0.elapsed().as_secs_f64());
    }
    let t0 = std::time::Instant::now();
    let (out, state) = steps::run(&fe, &RunOptions::default())?;
    if crate::settings::with(|s| s.timing) {
        eprintln!("steps::run: {:.3} s", t0.elapsed().as_secs_f64());
    }
    let fr = out.steps[0].frames.last().ok_or_else(|| FemError("intern: steget gav ingen frame".into()))?;

    let nn = m.nodes.len();
    let mut disp = vec![[0.0; 6]; nn];
    let mut reaction = vec![[0.0; 6]; nn];
    let mut fixed = vec![[false; 6]; nn];
    for n in 0..nn {
        for d in 0..6 {
            disp[n][d] = state.u[n][d];
            reaction[n][d] = fr.rf[n][d];
        }
    }
    for &(n, d, _) in &cons {
        fixed[n][d] = true;
    }
    let mut applied = [0.0; 6];
    for f in &fext {
        for d in 0..6 {
            applied[d] += f[d];
        }
    }
    let labels = ["S11", "S22", "S33", "S12", "S13", "S23"];
    let stress: Vec<Vec<[f64; 6]>> = fr
        .elem
        .iter()
        .map(|fields| {
            let Some(s) = fields.iter().find(|f| f.name == "S") else { return Vec::new() };
            let idx: Vec<Option<usize>> = s.labels.iter().map(|l| labels.iter().position(|x| x == l)).collect();
            s.points
                .iter()
                .map(|(_, _, v)| {
                    let mut out = [0.0; 6];
                    for (k, x) in v.iter().enumerate() {
                        if let Some(i) = idx[k] {
                            out[i] = *x;
                        }
                    }
                    out
                })
                .collect()
        })
        .collect();
    let mut beam_forces = vec![None; m.elements.len()];
    if m.elements.iter().any(|e| m.elem_kind(e).is_beam()) {
        let prep = Prepared::new(&fe)?;
        for (e, slot) in beam_forces.iter_mut().enumerate() {
            let beam = match fe.elements[e].etype.name() {
                "B23" => &beam_eb::B23,
                "B33" => &beam_eb::B33,
                _ => continue,
            };
            let ctx = prep.ctx(e, false, 0.0, 0.0);
            *slot = Some(beam.end_forces(&ctx, &prep.gather(e, &state.u))?);
        }
    }
    Ok(Results { disp, reaction, stress, beam_forces, active, fixed, applied, n_eq: out.n_eq })
}
