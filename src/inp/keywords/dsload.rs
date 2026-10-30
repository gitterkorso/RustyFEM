//! *DSLOAD: fördelad last på en elementyta (*SURFACE, TYPE=ELEMENT).
//!
//! Lasttypen P (normaltryck) översätts till elementlasten Pn för varje
//! elementyta Sn i ytan, så att elementens `distributed_load` används precis
//! som för *DLOAD. Andra lasttyper (TRVEC, TRSHR, ...) ger fel.

use crate::fem::model::{parse_f64, DistLoad, DloadBlock, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "DSLOAD", kind: Kind::Step, handler: dsload }];

fn dsload(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("FOLLOWER") || kw.has("CONSTANT RESULTANT") {
        return kw.err("FOLLOWER/CONSTANT RESULTANT stöds inte ännu");
    }
    let mut block = DloadBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        amplitude: kw.param("AMPLITUDE").map(|s| s.to_uppercase()),
        items: Vec::new(),
        source: kw.location(),
    };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'yta, lasttyp, storlek'");
        }
        let name = d.fields[0].trim().to_uppercase();
        let kind = d.fields[1].trim().to_uppercase();
        if kind != "P" {
            return kw.err_at(d.line, format!("lasttypen {kind} i *DSLOAD stöds inte ännu (bara P)"));
        }
        let Some(mag) = parse_f64(&d.fields[2]) else {
            return kw.err_at(d.line, format!("'{}' är inget tal", d.fields[2]));
        };
        let Some(sf) = ctx.model.surfaces.get(&name) else {
            return kw.err_at(d.line, format!("ytan {name} finns inte"));
        };
        if sf.kind != "ELEMENT" {
            return kw.err_at(d.line, format!("ytan {name} är inte en elementyta"));
        }
        for (e, face) in &sf.faces {
            let Some(num) = face.strip_prefix('S').and_then(|s| s.parse::<usize>().ok()) else {
                return kw.err_at(d.line, format!("ytbeteckningen {face} i {name} stöds inte för *DSLOAD ännu"));
            };
            block.items.push(DistLoad { elem: *e, kind: format!("P{num}"), magnitude: mag, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}
