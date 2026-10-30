//! Gränssnitt för nyckelordshanterare och läskontexten.
//!
//! En hanterare får hela nyckelordsblocket (`RawKeyword`: parametrar och
//! datarader) och uppdaterar `InpCtx`: modellen, parten som definieras,
//! aktuellt material eller aktuellt steg.

use crate::fem::model::{FeModel, MaterialDef, RawKeyword, StepDef, ASSEMBLY_INSTANCE, FLAT_INSTANCE};
use crate::model::{fail, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Var ett nyckelord får stå.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Modelldata (före första *STEP).
    Model,
    /// Historiedata (inne i *STEP ... *END STEP).
    Step,
    /// Både och.
    Both,
    /// Materialbeteende direkt efter *MATERIAL (eller ett annat beteende).
    Behavior,
}

pub type Handler = fn(&mut InpCtx, &RawKeyword) -> Result<()>;

/// En registrerad nyckelordshanterare.
pub struct Keyword {
    /// Nyckelordet utan '*', versaler, enkla mellanslag ("SOLID SECTION").
    pub name: &'static str,
    pub kind: Kind,
    pub handler: Handler,
}

/// Tillstånd medan .inp-filen läses.
#[derive(Default)]
pub struct InpCtx {
    pub model: FeModel,
    /// Färdiga parter (namn → geometri, set och tvärsnitt).
    pub parts: BTreeMap<String, FeModel>,
    /// Part som definieras just nu.
    pub part: Option<(String, FeModel)>,
    /// *ASSEMBLY har förekommit (noder utan instans hör då till assemblyn).
    pub assembly: bool,
    pub in_assembly: bool,
    /// Instans inom *INSTANCE ... *END INSTANCE.
    pub instance: Option<String>,
    /// Material vars beteenden läses just nu.
    pub material: Option<usize>,
    /// Steg som definieras just nu.
    pub step: Option<StepDef>,
    pub warnings: Vec<String>,
    /// Inkluderade filer (för att kunna kopiera dem tillsammans med indatafilen).
    pub included: Vec<PathBuf>,
    /// Katalogen för filen som läses (för relativa FILE=-referenser).
    pub dir: Option<PathBuf>,
}

/// Tolkar ett heltal ("12", "12.").
pub fn parse_label(s: &str) -> Option<i64> {
    let t = s.trim();
    t.parse::<i64>().ok().or_else(|| t.strip_suffix('.').and_then(|u| u.parse().ok()))
}

impl InpCtx {
    /// Modellen som geometri läggs i: parten som definieras eller huvudmodellen.
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

    /// Instansindex för noder/element som definieras nu.
    pub fn mesh_instance(&mut self) -> u32 {
        let name = match &self.part {
            Some((n, _)) => n.clone(),
            None if self.assembly => ASSEMBLY_INSTANCE.to_string(),
            None => FLAT_INSTANCE.to_string(),
        };
        self.mesh().instance(&name)
    }

    /// Instans som heltalsnummer i historiedata syftar på.
    pub fn default_instance(&self) -> &'static str {
        if self.assembly { ASSEMBLY_INSTANCE } else { FLAT_INSTANCE }
    }

    pub fn step_mut(&mut self, kw: &RawKeyword) -> Result<&mut StepDef> {
        match &mut self.step {
            Some(s) => Ok(s),
            None => kw.err("får bara stå inne i *STEP ... *END STEP"),
        }
    }

    pub fn material_mut(&mut self, kw: &RawKeyword) -> Result<&mut MaterialDef> {
        match self.material {
            Some(i) => Ok(&mut self.model.materials[i]),
            None => kw.err("materialbeteenden ska stå direkt efter *MATERIAL"),
        }
    }

    pub fn warn(&mut self, kw: &RawKeyword, msg: impl AsRef<str>) {
        self.warnings.push(format!("{}: {}", kw.location(), msg.as_ref()));
    }

    /// Noder som en referens på en datarad syftar på: nodnummer, "INSTANS.nummer"
    /// eller ett nodset ("SET" eller "INSTANS.SET"). `inst` = instans för nummer
    /// (från INSTANCE= eller den som definieras).
    pub fn node_ref(&self, kw: &RawKeyword, line: usize, token: &str, inst: Option<&str>) -> Result<Vec<usize>> {
        let tok = token.trim();
        let default = inst.map(|s| s.to_string()).unwrap_or_else(|| match &self.part {
            Some((n, _)) => n.clone(),
            None => self.default_instance().to_string(),
        });
        let mesh = self.mesh_ref();
        if let Some(l) = parse_label(tok) {
            let Some(i) = mesh.instance_index(&default) else {
                return kw.err_at(line, format!("instansen '{default}' finns inte"));
            };
            return match mesh.node(i, l) {
                Some(n) => Ok(vec![n]),
                None => kw.err_at(line, format!("nod {l} finns inte")),
            };
        }
        let up = tok.to_uppercase();
        if let Some((a, b)) = up.rsplit_once('.') {
            if let (Some(l), Some(i)) = (parse_label(b), mesh.instance_index(a)) {
                return match mesh.node(i, l) {
                    Some(n) => Ok(vec![n]),
                    None => kw.err_at(line, format!("nod {tok} finns inte")),
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
        kw.err_at(line, format!("nodsetet '{tok}' finns inte"))
    }

    /// Element som en referens syftar på (nummer, "INSTANS.nummer" eller elementset).
    pub fn elem_ref(&self, kw: &RawKeyword, line: usize, token: &str, inst: Option<&str>) -> Result<Vec<usize>> {
        let tok = token.trim();
        let default = inst.map(|s| s.to_string()).unwrap_or_else(|| match &self.part {
            Some((n, _)) => n.clone(),
            None => self.default_instance().to_string(),
        });
        let mesh = self.mesh_ref();
        if let Some(l) = parse_label(tok) {
            let Some(i) = mesh.instance_index(&default) else {
                return kw.err_at(line, format!("instansen '{default}' finns inte"));
            };
            return match mesh.element(i, l) {
                Some(e) => Ok(vec![e]),
                None => kw.err_at(line, format!("element {l} finns inte")),
            };
        }
        let up = tok.to_uppercase();
        if let Some((a, b)) = up.rsplit_once('.') {
            if let (Some(l), Some(i)) = (parse_label(b), mesh.instance_index(a)) {
                return match mesh.element(i, l) {
                    Some(e) => Ok(vec![e]),
                    None => kw.err_at(line, format!("element {tok} finns inte")),
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
        kw.err_at(line, format!("elementsetet '{tok}' finns inte"))
    }
}

/// Hanterare som sparar nyckelordet i `FeModel::extra` (för moduler som
/// läser det senare, t.ex. *TIE, *SURFACE, *AMPLITUDE).
pub fn store_model(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.mesh().extra.push(kw.clone());
    Ok(())
}

/// Hanterare som sparar nyckelordet i aktuellt stegs `extra`.
pub fn store_step(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.step_mut(kw)?.extra.push(kw.clone());
    Ok(())
}

/// Hanterare för procedurnyckelord (*STATIC, *FREQUENCY, ...): sätter stegets procedur.
pub fn procedure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let step = ctx.step_mut(kw)?;
    if let Some(p) = &step.procedure {
        return kw.err(format!("steget har redan proceduren *{}", p.name));
    }
    step.procedure = Some(kw.clone());
    Ok(())
}

/// Hanterare som godtar och ignorerar nyckelordet.
pub fn ignore(_ctx: &mut InpCtx, _kw: &RawKeyword) -> Result<()> {
    Ok(())
}

/// Hjälp: fel om ett nyckelord saknar en obligatorisk parameter.
pub fn required<'a>(kw: &'a RawKeyword, key: &str) -> Result<&'a str> {
    match kw.param(key) {
        Some(v) if !v.trim().is_empty() => Ok(v.trim()),
        _ => fail(format!("{}: parametern {key}= saknas", kw.location())),
    }
}
