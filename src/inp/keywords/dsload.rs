//! *DSLOAD: distributed load on an element-based surface (*SURFACE, TYPE=ELEMENT).
//!
//! The load type P (normal pressure) is translated to the element load Pn for each
//! element face Sn in the surface, so that the elements' `distributed_load` is used exactly
//! as for *DLOAD. Other load types (TRVEC, TRSHR, ...) give an error.

use crate::fem::model::{parse_f64, DistLoad, DloadBlock, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "DSLOAD", kind: Kind::Step, handler: dsload }];

fn dsload(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("FOLLOWER") || kw.has("CONSTANT RESULTANT") {
        return kw.err("FOLLOWER/CONSTANT RESULTANT is not supported yet");
    }
    let mut block = DloadBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        amplitude: kw.param("AMPLITUDE").map(|s| s.to_uppercase()),
        items: Vec::new(),
        source: kw.location(),
    };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'surface, load type, magnitude'");
        }
        let name = d.fields[0].trim().to_uppercase();
        let kind = d.fields[1].trim().to_uppercase();
        if kind != "P" {
            return kw.err_at(d.line, format!("load type {kind} in *DSLOAD is not supported yet (only P)"));
        }
        let Some(mag) = parse_f64(&d.fields[2]) else {
            return kw.err_at(d.line, format!("'{}' is not a number", d.fields[2]));
        };
        let Some(sf) = ctx.model.surfaces.get(&name) else {
            return kw.err_at(d.line, format!("surface {name} does not exist"));
        };
        if sf.kind != "ELEMENT" {
            return kw.err_at(d.line, format!("surface {name} is not an element-based surface"));
        }
        for (e, face) in &sf.faces {
            let Some(num) = face.strip_prefix('S').and_then(|s| s.parse::<usize>().ok()) else {
                return kw.err_at(d.line, format!("face identifier {face} in {name} is not supported for *DSLOAD yet"));
            };
            block.items.push(DistLoad { elem: *e, kind: format!("P{num}"), magnitude: mag, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}
