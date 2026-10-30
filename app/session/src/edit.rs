//! Edits as commands with an inverse (undo/redo).
//!
//! A `.rusty` session is edited as script text (`FemText`); the tools in the
//! UI write lines in the script. An `.inp` session is edited directly in
//! `FeModel`: values in material behaviors, sections, steps, boundary conditions and
//! loads. `apply` performs the edit and returns the command that reverts it.

use rustyfem::fem::dof;
use rustyfem::fem::model::{FeModel, SectionKind};
use serde::{Deserialize, Serialize};
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
#[serde(tag = "kind")]
pub enum Edit {
    /// The whole `.rusty` script.
    FemText { text: String },
    /// A data line in a material behavior (*ELASTIC, *PLASTIC, ...).
    MaterialData { material: u32, behavior: u32, line: u32, fields: Vec<String> },
    Density { material: u32, value: f64 },
    /// Thickness (planar elements, shells) or area (trusses).
    SectionThickness { section: u32, value: f64 },
    /// A data line in the step's procedure (e.g. *STATIC: increment, time, min, max).
    ProcedureData { step: u32, line: u32, fields: Vec<String> },
    Nlgeom { step: u32, value: Option<bool> },
    MaxIncrements { step: u32, value: u32 },
    /// Value in a *BOUNDARY block. `step` = null for model data.
    BcValue { step: Option<u32>, block: u32, item: u32, value: f64 },
    CloadValue { step: u32, block: u32, item: u32, value: f64 },
    DloadValue { step: u32, block: u32, item: u32, value: f64 },
    /// Renames a step.
    StepName { step: u32, name: String },
}

impl Edit {
    /// Short description for the undo/redo menu.
    pub fn label(&self) -> String {
        match self {
            Edit::FemText { .. } => "edit model".into(),
            Edit::MaterialData { .. } => "edit material".into(),
            Edit::Density { .. } => "edit density".into(),
            Edit::SectionThickness { .. } => "edit section".into(),
            Edit::ProcedureData { .. } => "edit step times".into(),
            Edit::Nlgeom { .. } => "edit NLGEOM".into(),
            Edit::MaxIncrements { .. } => "edit number of increments".into(),
            Edit::BcValue { .. } => "edit boundary condition".into(),
            Edit::CloadValue { .. } => "edit point load".into(),
            Edit::DloadValue { .. } => "edit distributed load".into(),
            Edit::StepName { .. } => "rename step".into(),
        }
    }
}

fn get<'a, T>(v: &'a mut [T], i: u32, what: &str) -> Result<&'a mut T, String> {
    v.get_mut(i as usize).ok_or_else(|| format!("{what} {} does not exist", i + 1))
}

fn step<'a>(m: &'a mut FeModel, s: u32) -> Result<&'a mut rustyfem::fem::model::StepDef, String> {
    get(&mut m.steps, s, "step")
}

/// Performs an edit in the model. `FemText` is handled by the session.
pub fn apply(m: &mut FeModel, e: &Edit) -> Result<Edit, String> {
    Ok(match e {
        Edit::FemText { .. } => return Err("internal: FemText is handled by the session".into()),
        Edit::MaterialData { material, behavior, line, fields } => {
            let mat = get(&mut m.materials, *material, "material")?;
            let b = get(&mut mat.behaviors, *behavior, "behavior")?;
            for f in fields {
                if rustyfem::fem::model::parse_f64(f).is_none() {
                    return Err(format!("'{f}' is not a number"));
                }
            }
            let d = get(&mut b.data, *line, "line")?;
            let old = std::mem::replace(&mut d.fields, fields.clone());
            Edit::MaterialData { material: *material, behavior: *behavior, line: *line, fields: old }
        }
        Edit::Density { material, value } => {
            let mat = get(&mut m.materials, *material, "material")?;
            let old = std::mem::replace(&mut mat.density, *value);
            Edit::Density { material: *material, value: old }
        }
        Edit::SectionThickness { section, value } => {
            if !(*value > 0.0) {
                return Err("the thickness must be positive".into());
            }
            let s = get(&mut m.sections, *section, "section")?;
            let old = match s.kind {
                SectionKind::Solid { ref mut thickness } => thickness.replace(*value).unwrap_or(1.0),
                SectionKind::Shell { ref mut thickness, .. } => std::mem::replace(thickness, *value),
                _ => return Err("the section has no thickness".into()),
            };
            if let Some(d) = s.raw.data.first_mut() {
                match d.fields.first_mut() {
                    Some(f) => *f = format!("{value:e}"),
                    None => d.fields.push(format!("{value:e}")),
                }
            }
            Edit::SectionThickness { section: *section, value: old }
        }
        Edit::ProcedureData { step: s, line, fields } => {
            for f in fields {
                if rustyfem::fem::model::parse_f64(f).is_none() {
                    return Err(format!("'{f}' is not a number"));
                }
            }
            let st = step(m, *s)?;
            let p = st.procedure.as_mut().ok_or("the step has no procedure")?;
            while p.data.len() <= *line as usize {
                p.data.push(Default::default());
            }
            let old = std::mem::replace(&mut p.data[*line as usize].fields, fields.clone());
            Edit::ProcedureData { step: *s, line: *line, fields: old }
        }
        Edit::Nlgeom { step: s, value } => {
            let st = step(m, *s)?;
            let old = std::mem::replace(&mut st.nlgeom, *value);
            Edit::Nlgeom { step: *s, value: old }
        }
        Edit::MaxIncrements { step: s, value } => {
            let st = step(m, *s)?;
            let old = std::mem::replace(&mut st.max_increments, (*value).max(1) as usize);
            Edit::MaxIncrements { step: *s, value: old as u32 }
        }
        Edit::BcValue { step: s, block, item, value } => {
            let blocks = match s {
                Some(s) => &mut step(m, *s)?.bcs,
                None => &mut m.initial_bcs,
            };
            let b = get(blocks, *block, "boundary condition")?;
            let it = get(&mut b.items, *item, "line")?;
            let old = std::mem::replace(&mut it.2, *value);
            Edit::BcValue { step: *s, block: *block, item: *item, value: old }
        }
        Edit::CloadValue { step: s, block, item, value } => {
            let b = get(&mut step(m, *s)?.cloads, *block, "point load")?;
            let it = get(&mut b.items, *item, "line")?;
            let old = std::mem::replace(&mut it.2, *value);
            Edit::CloadValue { step: *s, block: *block, item: *item, value: old }
        }
        Edit::DloadValue { step: s, block, item, value } => {
            let b = get(&mut step(m, *s)?.dloads, *block, "distributed load")?;
            let it = get(&mut b.items, *item, "line")?;
            let old = std::mem::replace(&mut it.magnitude, *value);
            Edit::DloadValue { step: *s, block: *block, item: *item, value: old }
        }
        Edit::StepName { step: s, name } => {
            let st = step(m, *s)?;
            let old = std::mem::replace(&mut st.name, name.trim().to_uppercase());
            Edit::StepName { step: *s, name: old }
        }
    })
}

/// Name of a degree of freedom for display.
pub fn dof_name(d: dof::Dof) -> &'static str {
    dof::name(d)
}
