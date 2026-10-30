//! Reader for input files in .inp format.
//!
//! The file is split into keyword blocks (`reader.rs`). Each keyword is handled
//! by a registered handler (`keywords/`), so new keywords are added in
//! their own files without changing the reader. Unknown keywords give a warning, or
//! an error if they are in the list of keywords that affect the result
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

/// A loaded model with warnings and included files.
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
                panic!("keyword *{} is registered twice", k.name);
            }
        }
        m
    })
}

/// The handler for a keyword (upper case, without '*').
pub fn lookup(name: &str) -> Option<&'static Keyword> {
    table().get(name).copied()
}

/// All registered keywords, sorted.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&str> = table().keys().copied().collect();
    v.sort_unstable();
    v
}

fn dispatch(ctx: &mut InpCtx, blocks: Vec<crate::fem::model::RawKeyword>) -> Result<()> {
    // *PARAMETER and <name> (parameters.rs)
    let blocks = parameters::expand(blocks)?;
    for kw in &blocks {
        let Some(k) = lookup(&kw.name) else {
            if unsupported::AFFECTS_RESULTS.contains(&kw.name.as_str()) {
                return kw.err("the keyword is not supported yet and affects the result");
            }
            ctx.warn(kw, "unknown keyword ignored");
            continue;
        };
        // Area cavity: *DAMAGE INITIATION/EVOLUTION/STABILIZATION under *COHESIVE BEHAVIOR (cohesive contact)
        if k.kind == Kind::Behavior && ctx.material.is_none() && keywords::cohesive_contact::surface_damage(ctx, kw)? {
            continue;
        }
        if k.kind != Kind::Behavior {
            ctx.material = None;
        }
        match k.kind {
            Kind::Model if ctx.step.is_some() => return kw.err("model data may not appear inside a step"),
            Kind::Step if ctx.step.is_none() => return kw.err("may only appear inside *STEP ... *END STEP"),
            Kind::Behavior if ctx.material.is_none() => return kw.err("material behaviors must appear directly after *MATERIAL"),
            _ => {}
        }
        (k.handler)(ctx, kw)?;
    }
    if ctx.step.is_some() {
        return fail("the file ends inside a step – *END STEP is missing");
    }
    if let Some((name, _)) = &ctx.part {
        return fail(format!("part {name} lacks *END PART"));
    }
    ctx.model.warnings.extend(ctx.warnings.iter().cloned());
    ctx.model.finalize()?;
    // Area elements3: the generated plane nodes of the SAXA elements (node + 100000·k)
    crate::elements::elements3::saxa::setup(&mut ctx.model)?;
    // Area beams3: the elbows' ovalization nodes and NOWARP/NOOVAL/NODEFORM
    crate::elements::elbow::setup(&mut ctx.model)?;
    // *RIGID BODY, POSITION=CENTER OF MASS
    crate::interactions::rigid_body::reposition(&mut ctx.model)?;
    // *CONTACT PAIR, ADJUST= and *CONTACT INITIALIZATION DATA (Area contact3);
    // before the normals since the slave nodes are moved
    crate::interactions::contact::adjust_slaves(&mut ctx.model)?;
    // Area hybrid2: the pressure nodes of C3D4H (one per corner node and material, elements/hybrid_tet.rs)
    crate::elements::hybrid_tet::setup(&mut ctx.model)?;
    // *ENRICHMENT: XFEM elements and enrichment nodes (elements/xfem.rs)
    crate::elements::xfem::setup(&mut ctx.model)?;
    // Area explicit_adv: continuum elements that are converted to SPH particles
    crate::steps::explicit::adv::conversion::apply(&mut ctx.model)?;
    // Area normals: averaged nodal normals and *NORMAL
    crate::elements::normals::apply(&mut ctx.model)
}

/// Reads an .inp file (with *INCLUDE).
pub fn read_file(path: &Path) -> Result<InpModel> {
    let mut ctx = InpCtx::default();
    ctx.dir = path.parent().map(|p| p.to_path_buf());
    let blocks = reader::blocks_from_file(path, &mut ctx.included, 0)?;
    dispatch(&mut ctx, blocks)?;
    Ok(InpModel { warnings: ctx.warnings.clone(), included: ctx.included.clone(), model: ctx.model })
}

/// Reads .inp text. Relative *INCLUDE paths are resolved from `dir`.
pub fn read_str(text: &str, dir: &Path) -> Result<InpModel> {
    read_str_with(text, dir, &reader::fs_resolver)
}

/// Reads .inp text with the *INCLUDE files from a file resolver (e.g. the app's
/// document folder or a project file instead of the file system).
pub fn read_str_with(text: &str, dir: &Path, resolve: reader::Resolver) -> Result<InpModel> {
    let mut ctx = InpCtx::default();
    ctx.dir = Some(dir.to_path_buf());
    let blocks = reader::blocks_from_str_with(text, "(text)", dir, resolve, &mut ctx.included, 0)?;
    dispatch(&mut ctx, blocks)?;
    Ok(InpModel { warnings: ctx.warnings.clone(), included: ctx.included.clone(), model: ctx.model })
}
