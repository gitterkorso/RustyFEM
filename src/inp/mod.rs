//! Läsare för indatafiler (.inp).
//!
//! Filen delas upp i nyckelordsblock (`reader.rs`). Varje nyckelord hanteras
//! av en registrerad hanterare (`keywords/`), så nya nyckelord läggs till i
//! egna filer utan att ändra läsaren. Okända nyckelord ger en varning, eller
//! ett fel om de finns i listan över nyckelord som påverkar resultatet
//! (`unsupported.rs`).

mod api;
pub mod keywords;
pub mod parameters;
pub mod reader;
pub mod unsupported;
pub use api::*;

use crate::fem::model::FeModel;
use crate::model::{fail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// En inläst modell med varningar och inkluderade filer.
pub struct InpModel {
    pub model: FeModel,
    pub warnings: Vec<String>,
    pub included: Vec<PathBuf>,
}

fn table() -> &'static HashMap<&'static str, &'static Keyword> {
    static T: OnceLock<HashMap<&'static str, &'static Keyword>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = HashMap::new();
        for k in keywords::all() {
            if m.insert(k.name, k).is_some() {
                panic!("nyckelordet *{} är registrerat två gånger", k.name);
            }
        }
        m
    })
}

/// Hanteraren för ett nyckelord (versaler, utan '*').
pub fn lookup(name: &str) -> Option<&'static Keyword> {
    table().get(name).copied()
}

/// Alla registrerade nyckelord, sorterade.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&str> = table().keys().copied().collect();
    v.sort_unstable();
    v
}

fn dispatch(ctx: &mut InpCtx, blocks: Vec<crate::fem::model::RawKeyword>) -> Result<()> {
    // *PARAMETER och <namn> (parameters.rs)
    let blocks = parameters::expand(blocks)?;
    for kw in &blocks {
        let Some(k) = lookup(&kw.name) else {
            if unsupported::AFFECTS_RESULTS.contains(&kw.name.as_str()) {
                return kw.err("nyckelordet stöds inte ännu och påverkar resultatet");
            }
            ctx.warn(kw, "okänt nyckelord ignoreras");
            continue;
        };
        // område cavity: *DAMAGE INITIATION/EVOLUTION/STABILIZATION under *COHESIVE BEHAVIOR (kohesiv kontakt)
        if k.kind == Kind::Behavior && ctx.material.is_none() && keywords::cohesive_contact::surface_damage(ctx, kw)? {
            continue;
        }
        if k.kind != Kind::Behavior {
            ctx.material = None;
        }
        match k.kind {
            Kind::Model if ctx.step.is_some() => return kw.err("modelldata får inte stå inne i ett steg"),
            Kind::Step if ctx.step.is_none() => return kw.err("får bara stå inne i *STEP ... *END STEP"),
            Kind::Behavior if ctx.material.is_none() => return kw.err("materialbeteenden ska stå direkt efter *MATERIAL"),
            _ => {}
        }
        (k.handler)(ctx, kw)?;
    }
    if ctx.step.is_some() {
        return fail("filen slutar inne i ett steg – *END STEP saknas");
    }
    if let Some((name, _)) = &ctx.part {
        return fail(format!("parten {name} saknar *END PART"));
    }
    ctx.model.warnings.extend(ctx.warnings.iter().cloned());
    ctx.model.finalize()?;
    // område elements3: SAXA-elementens genererade plannoder (nod + 100000·k)
    crate::elements::elements3::saxa::setup(&mut ctx.model)?;
    // område beams3: rörkrökarnas ovaliseringsnoder och NOWARP/NOOVAL/NODEFORM
    crate::elements::elbow::setup(&mut ctx.model)?;
    // *RIGID BODY, POSITION=CENTER OF MASS
    crate::interactions::rigid_body::reposition(&mut ctx.model)?;
    // *CONTACT PAIR, ADJUST= och *CONTACT INITIALIZATION DATA (område contact3);
    // före normalerna eftersom slavnoderna flyttas
    crate::interactions::contact::adjust_slaves(&mut ctx.model)?;
    // område hybrid2: C3D4H:s trycknoder (en per hörnnod och material, elements/hybrid_tet.rs)
    crate::elements::hybrid_tet::setup(&mut ctx.model)?;
    // *ENRICHMENT: XFEM-element och berikningsnoder (elements/xfem.rs)
    crate::elements::xfem::setup(&mut ctx.model)?;
    // område explicit_adv: kontinuumelement som omvandlas till SPH-partiklar
    crate::steps::explicit::adv::conversion::apply(&mut ctx.model)?;
    // område normals: medelvärdesbildade nodnormaler och *NORMAL
    crate::elements::normals::apply(&mut ctx.model)
}

/// Läser en .inp-fil (med *INCLUDE).
pub fn read_file(path: &Path) -> Result<InpModel> {
    let mut ctx = InpCtx::default();
    ctx.dir = path.parent().map(|p| p.to_path_buf());
    let blocks = reader::blocks_from_file(path, &mut ctx.included, 0)?;
    dispatch(&mut ctx, blocks)?;
    Ok(InpModel { warnings: ctx.warnings.clone(), included: ctx.included.clone(), model: ctx.model })
}

/// Läser .inp-text. Relativa *INCLUDE tolkas från `dir`.
pub fn read_str(text: &str, dir: &Path) -> Result<InpModel> {
    read_str_with(text, dir, &reader::fs_resolver)
}

/// Läser .inp-text med *INCLUDE-filerna från en filresolver (t.ex. appens
/// dokumentmapp eller en projektfil i stället för filsystemet).
pub fn read_str_with(text: &str, dir: &Path, resolve: reader::Resolver) -> Result<InpModel> {
    let mut ctx = InpCtx::default();
    ctx.dir = Some(dir.to_path_buf());
    let blocks = reader::blocks_from_str_with(text, "(text)", dir, resolve, &mut ctx.included, 0)?;
    dispatch(&mut ctx, blocks)?;
    Ok(InpModel { warnings: ctx.warnings.clone(), included: ctx.included.clone(), model: ctx.model })
}
