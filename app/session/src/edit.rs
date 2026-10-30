//! Redigeringar som kommandon med invers (ångra/gör om).
//!
//! En `.rusty`-session redigeras som skripttext (`FemText`); verktygen i
//! gränssnittet skriver rader i skriptet. En `.inp`-session redigeras direkt i
//! `FeModel`: värden i materialbeteenden, tvärsnitt, steg, randvillkor och
//! laster. `apply` utför ändringen och returnerar kommandot som återställer den.

use rustyfem::fem::dof;
use rustyfem::fem::model::{FeModel, SectionKind};
use serde::{Deserialize, Serialize};
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
#[serde(tag = "kind")]
pub enum Edit {
    /// Hela `.rusty`-skriptet.
    FemText { text: String },
    /// En datarad i ett materialbeteende (*ELASTIC, *PLASTIC, ...).
    MaterialData { material: u32, behavior: u32, line: u32, fields: Vec<String> },
    Density { material: u32, value: f64 },
    /// Tjocklek (plana element, skal) eller area (stänger).
    SectionThickness { section: u32, value: f64 },
    /// En datarad i stegets procedur (t.ex. *STATIC: inkrement, tid, min, max).
    ProcedureData { step: u32, line: u32, fields: Vec<String> },
    Nlgeom { step: u32, value: Option<bool> },
    MaxIncrements { step: u32, value: u32 },
    /// Värde i ett *BOUNDARY-block. `step` = null för modelldata.
    BcValue { step: Option<u32>, block: u32, item: u32, value: f64 },
    CloadValue { step: u32, block: u32, item: u32, value: f64 },
    DloadValue { step: u32, block: u32, item: u32, value: f64 },
    /// Byter namn på ett steg.
    StepName { step: u32, name: String },
}

impl Edit {
    /// Kort beskrivning för ångra/gör om-menyn.
    pub fn label(&self) -> String {
        match self {
            Edit::FemText { .. } => "ändra modell".into(),
            Edit::MaterialData { .. } => "ändra material".into(),
            Edit::Density { .. } => "ändra densitet".into(),
            Edit::SectionThickness { .. } => "ändra tvärsnitt".into(),
            Edit::ProcedureData { .. } => "ändra stegets tider".into(),
            Edit::Nlgeom { .. } => "ändra NLGEOM".into(),
            Edit::MaxIncrements { .. } => "ändra antal inkrement".into(),
            Edit::BcValue { .. } => "ändra randvillkor".into(),
            Edit::CloadValue { .. } => "ändra punktlast".into(),
            Edit::DloadValue { .. } => "ändra utbredd last".into(),
            Edit::StepName { .. } => "byt namn på steg".into(),
        }
    }
}

fn get<'a, T>(v: &'a mut [T], i: u32, what: &str) -> Result<&'a mut T, String> {
    v.get_mut(i as usize).ok_or_else(|| format!("{what} {} finns inte", i + 1))
}

fn step<'a>(m: &'a mut FeModel, s: u32) -> Result<&'a mut rustyfem::fem::model::StepDef, String> {
    get(&mut m.steps, s, "steg")
}

/// Utför en ändring i modellen. `FemText` hanteras av sessionen.
pub fn apply(m: &mut FeModel, e: &Edit) -> Result<Edit, String> {
    Ok(match e {
        Edit::FemText { .. } => return Err("intern: FemText hanteras av sessionen".into()),
        Edit::MaterialData { material, behavior, line, fields } => {
            let mat = get(&mut m.materials, *material, "material")?;
            let b = get(&mut mat.behaviors, *behavior, "beteende")?;
            for f in fields {
                if rustyfem::fem::model::parse_f64(f).is_none() {
                    return Err(format!("'{f}' är inget tal"));
                }
            }
            let d = get(&mut b.data, *line, "rad")?;
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
                return Err("tjockleken ska vara positiv".into());
            }
            let s = get(&mut m.sections, *section, "tvärsnitt")?;
            let old = match s.kind {
                SectionKind::Solid { ref mut thickness } => thickness.replace(*value).unwrap_or(1.0),
                SectionKind::Shell { ref mut thickness, .. } => std::mem::replace(thickness, *value),
                _ => return Err("tvärsnittet har ingen tjocklek".into()),
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
                    return Err(format!("'{f}' är inget tal"));
                }
            }
            let st = step(m, *s)?;
            let p = st.procedure.as_mut().ok_or("steget saknar procedur")?;
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
            let b = get(blocks, *block, "randvillkor")?;
            let it = get(&mut b.items, *item, "rad")?;
            let old = std::mem::replace(&mut it.2, *value);
            Edit::BcValue { step: *s, block: *block, item: *item, value: old }
        }
        Edit::CloadValue { step: s, block, item, value } => {
            let b = get(&mut step(m, *s)?.cloads, *block, "punktlast")?;
            let it = get(&mut b.items, *item, "rad")?;
            let old = std::mem::replace(&mut it.2, *value);
            Edit::CloadValue { step: *s, block: *block, item: *item, value: old }
        }
        Edit::DloadValue { step: s, block, item, value } => {
            let b = get(&mut step(m, *s)?.dloads, *block, "utbredd last")?;
            let it = get(&mut b.items, *item, "rad")?;
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

/// Namn på en frihetsgrad för visning.
pub fn dof_name(d: dof::Dof) -> &'static str {
    dof::name(d)
}
