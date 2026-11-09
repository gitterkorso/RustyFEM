//! *DSLOAD: distributed load on an element-based surface (*SURFACE, TYPE=ELEMENT).
//!
//! The load type P (normal pressure) is translated to the element load Pn for each
//! element face Sn in the surface, so that the elements' `distributed_load` is used exactly
//! as for *DLOAD. The surface tractions TRVEC (fixed direction) and TRSHR (shear:
//! the direction projected onto the surface) become the element loads TRVECn/TRSHRn
//! with the direction in `extra` (`elements/traction.rs`); on shells SPOS/SNEG is face 1.
//! Data line: surface, load type, magnitude[, d1, d2, d3].

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
        let traction = kind == "TRVEC" || kind == "TRSHR";
        if kind != "P" && !traction {
            return kw.err_at(d.line, format!("load type {kind} in *DSLOAD is not supported yet (only P, TRVEC and TRSHR)"));
        }
        let mut dir = Vec::new();
        if traction {
            for i in 3..6 {
                let f = d.fields.get(i).map(|f| f.trim()).unwrap_or("");
                match parse_f64(if f.is_empty() { "0" } else { f }) {
                    Some(v) => dir.push(v),
                    None => return kw.err_at(d.line, format!("'{f}' is not a number")),
                }
            }
            if dir.iter().all(|v| *v == 0.0) {
                return kw.err_at(d.line, format!("{kind} needs a direction: 'surface, {kind}, magnitude, d1, d2, d3'"));
            }
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
            let num = match face.as_str() {
                "SPOS" | "SNEG" if traction => Some(1),
                _ => face.strip_prefix('S').and_then(|s| s.parse::<usize>().ok()),
            };
            let Some(num) = num else {
                return kw.err_at(d.line, format!("face identifier {face} in {name} is not supported for *DSLOAD {kind} yet"));
            };
            let kind = if traction { format!("{kind}{num}") } else { format!("P{num}") };
            block.items.push(DistLoad { elem: *e, kind, magnitude: mag, extra: dir.clone() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}
