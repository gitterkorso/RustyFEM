//! *STEP, *END STEP and the procedure keyword *STATIC.
//!
//! Other procedures (*FREQUENCY, *BUCKLE, *DYNAMIC, *HEAT TRANSFER, ...)
//! are registered in their own files with the handler `inp::procedure`.

use crate::fem::model::{RawKeyword, StepDef};
use crate::inp::{parse_label, procedure, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "END STEP", kind: Kind::Step, handler: end_step },
    Keyword { name: "STATIC", kind: Kind::Step, handler: procedure },
    Keyword { name: "STEP", kind: Kind::Model, handler: step },
];

fn step(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() || ctx.in_assembly {
        return kw.err("*STEP may not appear inside *PART or *ASSEMBLY");
    }
    let index = ctx.model.steps.len() + 1;
    let name = kw.param("NAME").map(|s| s.trim().to_string()).unwrap_or_else(|| format!("Step-{index}"));
    let nlgeom = if kw.has("NLGEOM") { Some(kw.flag("NLGEOM", false)?) } else { None };
    let max_increments = match kw.param("INC") {
        Some(v) => parse_label(v).filter(|&n| n > 0).ok_or_else(|| crate::model::FemError(format!("{}: INC={v} must be a positive integer", kw.location())))? as usize,
        None => 100,
    };
    let amplitude_step = match kw.param_upper("AMPLITUDE") {
        None => false,
        Some(a) if a == "RAMP" => false,
        Some(a) if a == "STEP" => true,
        Some(a) => return kw.err(format!("AMPLITUDE={a} must be RAMP or STEP")),
    };
    for p in ["CONVERT SDI", "EXTRAPOLATION"] {
        if kw.has(p) {
            ctx.warn(kw, format!("parameter {p} is ignored"));
        }
    }
    ctx.step = Some(StepDef {
        name,
        header: kw.clone(),
        nlgeom,
        perturbation: kw.has("PERTURBATION"),
        max_increments,
        amplitude_step,
        ..Default::default()
    });
    Ok(())
}

fn end_step(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let s = ctx.step.take().unwrap();
    if s.procedure.is_none() {
        return kw.err(format!("step {} has no procedure (e.g. *STATIC)", s.name));
    }
    ctx.model.steps.push(s);
    Ok(())
}
