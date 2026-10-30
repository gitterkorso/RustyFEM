//! Interface for keyword handlers and the read context.
//!
//! A handler receives the whole keyword block (`RawKeyword`: parameters and
//! data lines) and updates `InpCtx`: the model, the part being defined,
//! the current material or the current step.

use crate::fem::model::{FeModel, MaterialDef, RawKeyword, StepDef, ASSEMBLY_INSTANCE, FLAT_INSTANCE};
use crate::model::{fail, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Where a keyword may appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Model data (before the first *STEP).
    Model,
    /// History data (inside *STEP ... *END STEP).
    Step,
    /// Both.
    Both,
    /// Material behavior directly after *MATERIAL (or another behavior).
    Behavior,
}

pub type Handler = fn(&mut InpCtx, &RawKeyword) -> Result<()>;

/// A registered keyword handler.
pub struct Keyword {
    /// The keyword without '*', upper case, single spaces ("SOLID SECTION").
    pub name: &'static str,
    pub kind: Kind,
    pub handler: Handler,
}

/// State while the .inp file is being read.
#[derive(Default)]
pub struct InpCtx {
    pub model: FeModel,
    /// Finished parts (name → geometry, sets and sections).
    pub parts: BTreeMap<String, FeModel>,
    /// Part currently being defined.
    pub part: Option<(String, FeModel)>,
    /// *ASSEMBLY has appeared (nodes without an instance then belong to the assembly).
    pub assembly: bool,
    pub in_assembly: bool,
    /// Instance inside *INSTANCE ... *END INSTANCE.
    pub instance: Option<String>,
    /// Material whose behaviors are currently being read.
    pub material: Option<usize>,
    /// Step currently being defined.
    pub step: Option<StepDef>,
    pub warnings: Vec<String>,
    /// Included files (so they can be copied together with the input file).
    pub included: Vec<PathBuf>,
    /// Directory of the file being read (for relative FILE= references).
    pub dir: Option<PathBuf>,
}

/// Parses an integer ("12", "12.").
pub fn parse_label(s: &str) -> Option<i64> {
    let t = s.trim();
    t.parse::<i64>().ok().or_else(|| t.strip_suffix('.').and_then(|u| u.parse().ok()))
}

impl InpCtx {
    /// The model geometry is added to: the part being defined or the main model.
    pub fn mesh(&mut self) -> &mut FeModel {
        match &mut self.part {
            Some((_, p)) => p,
            None => &mut self.model,
        }
    }

    pub fn mesh_ref(&self) -> &FeModel {
        match &self.part {
            Some((_, p)) => p,
            None => &self.model,
        }
    }

    /// Instance index for nodes/elements being defined now.
    pub fn mesh_instance(&mut self) -> u32 {
        let name = match &self.part {
            Some((n, _)) => n.clone(),
            None if self.assembly => ASSEMBLY_INSTANCE.to_string(),
            None => FLAT_INSTANCE.to_string(),
        };
        self.mesh().instance(&name)
    }

    /// Instance that integer labels in history data refer to.
    pub fn default_instance(&self) -> &'static str {
        if self.assembly { ASSEMBLY_INSTANCE } else { FLAT_INSTANCE }
    }

    pub fn step_mut(&mut self, kw: &RawKeyword) -> Result<&mut StepDef> {
        match &mut self.step {
            Some(s) => Ok(s),
            None => kw.err("may only appear inside *STEP ... *END STEP"),
        }
    }

    pub fn material_mut(&mut self, kw: &RawKeyword) -> Result<&mut MaterialDef> {
        match self.material {
            Some(i) => Ok(&mut self.model.materials[i]),
            None => kw.err("material behaviors must appear directly after *MATERIAL"),
        }
    }

    pub fn warn(&mut self, kw: &RawKeyword, msg: impl AsRef<str>) {
        self.warnings.push(format!("{}: {}", kw.location(), msg.as_ref()));
    }

    /// Nodes that a reference on a data line refers to: node label, "INSTANCE.label"
    /// or a node set ("SET" or "INSTANCE.SET"). `inst` = instance for labels
    /// (from INSTANCE= or the one being defined).
    pub fn node_ref(&self, kw: &RawKeyword, line: usize, token: &str, inst: Option<&str>) -> Result<Vec<usize>> {
        let tok = token.trim();
        let default = inst.map(|s| s.to_string()).unwrap_or_else(|| match &self.part {
            Some((n, _)) => n.clone(),
            None => self.default_instance().to_string(),
        });
        let mesh = self.mesh_ref();
        if let Some(l) = parse_label(tok) {
            let Some(i) = mesh.instance_index(&default) else {
                return kw.err_at(line, format!("instance '{default}' does not exist"));
            };
            return match mesh.node(i, l) {
                Some(n) => Ok(vec![n]),
                None => kw.err_at(line, format!("node {l} does not exist")),
            };
        }
        let up = tok.to_uppercase();
        if let Some((a, b)) = up.rsplit_once('.') {
            if let (Some(l), Some(i)) = (parse_label(b), mesh.instance_index(a)) {
                return match mesh.node(i, l) {
                    Some(n) => Ok(vec![n]),
                    None => kw.err_at(line, format!("node {tok} does not exist")),
                };
            }
        }
        let cands = match inst {
            Some(i) => vec![format!("{}.{up}", i.to_uppercase()), up.clone()],
            None => vec![up.clone()],
        };
        for c in &cands {
            if let Some(s) = mesh.nsets.get(c) {
                return Ok(s.clone());
            }
        }
        kw.err_at(line, format!("node set '{tok}' does not exist"))
    }

    /// Elements that a reference refers to (label, "INSTANCE.label" or element set).
    pub fn elem_ref(&self, kw: &RawKeyword, line: usize, token: &str, inst: Option<&str>) -> Result<Vec<usize>> {
        let tok = token.trim();
        let default = inst.map(|s| s.to_string()).unwrap_or_else(|| match &self.part {
            Some((n, _)) => n.clone(),
            None => self.default_instance().to_string(),
        });
        let mesh = self.mesh_ref();
        if let Some(l) = parse_label(tok) {
            let Some(i) = mesh.instance_index(&default) else {
                return kw.err_at(line, format!("instance '{default}' does not exist"));
            };
            return match mesh.element(i, l) {
                Some(e) => Ok(vec![e]),
                None => kw.err_at(line, format!("element {l} does not exist")),
            };
        }
        let up = tok.to_uppercase();
        if let Some((a, b)) = up.rsplit_once('.') {
            if let (Some(l), Some(i)) = (parse_label(b), mesh.instance_index(a)) {
                return match mesh.element(i, l) {
                    Some(e) => Ok(vec![e]),
                    None => kw.err_at(line, format!("element {tok} does not exist")),
                };
            }
        }
        let cands = match inst {
            Some(i) => vec![format!("{}.{up}", i.to_uppercase()), up.clone()],
            None => vec![up.clone()],
        };
        for c in &cands {
            if let Some(s) = mesh.elsets.get(c) {
                return Ok(s.clone());
            }
        }
        kw.err_at(line, format!("element set '{tok}' does not exist"))
    }
}

/// Handler that stores the keyword in `FeModel::extra` (for modules that
/// read it later, e.g. *TIE, *SURFACE, *AMPLITUDE).
pub fn store_model(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.mesh().extra.push(kw.clone());
    Ok(())
}

/// Handler that stores the keyword in the current step's `extra`.
pub fn store_step(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.step_mut(kw)?.extra.push(kw.clone());
    Ok(())
}

/// Handler for procedure keywords (*STATIC, *FREQUENCY, ...): sets the step's procedure.
pub fn procedure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let step = ctx.step_mut(kw)?;
    if let Some(p) = &step.procedure {
        return kw.err(format!("the step already has the procedure *{}", p.name));
    }
    step.procedure = Some(kw.clone());
    Ok(())
}

/// Handler that accepts and ignores the keyword.
pub fn ignore(_ctx: &mut InpCtx, _kw: &RawKeyword) -> Result<()> {
    Ok(())
}

/// Helper: error if a keyword lacks a required parameter.
pub fn required<'a>(kw: &'a RawKeyword, key: &str) -> Result<&'a str> {
    match kw.param(key) {
        Some(v) if !v.trim().is_empty() => Ok(v.trim()),
        _ => fail(format!("{}: parameter {key}= is missing", kw.location())),
    }
}
