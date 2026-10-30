//! *CLOAD, *DLOAD and *AMPLITUDE (stored for later use).

use crate::fem::dof::{self, Dof};
use crate::fem::model::{parse_f64, CloadBlock, DistLoad, DloadBlock, RawKeyword};
use crate::inp::{parse_label, store_model, InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "AMPLITUDE", kind: Kind::Model, handler: store_model },
    Keyword { name: "CLOAD", kind: Kind::Step, handler: cload },
    Keyword { name: "DLOAD", kind: Kind::Step, handler: dload },
];

fn num(kw: &RawKeyword, line: usize, s: &str) -> Result<f64> {
    parse_f64(s).ok_or_else(|| FemError(format!("{} line {line} (*{}): '{s}' is not a number", kw.file, kw.name)))
}

fn cload(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut block = CloadBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        amplitude: kw.param("AMPLITUDE").map(|s| s.to_uppercase()),
        follower: kw.flag("FOLLOWER", false)?,
        items: Vec::new(),
        source: kw.location(),
    };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'node/nset, degree of freedom, magnitude'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        let Some(k) = parse_label(&d.fields[1]).filter(|&k| k > 0 && dof::slot(k as Dof).is_some()) else {
            return kw.err_at(d.line, format!("'{}' is not a degree of freedom", d.fields[1]));
        };
        let v = num(kw, d.line, &d.fields[2])?;
        for n in nodes {
            block.items.push((n, k as Dof, v));
        }
    }
    let step = ctx.step_mut(kw)?;
    step.cloads.push(block);
    // *RANDOM RESPONSE: the load case number for the block (steps/random_response.rs)
    if let Some(lc) = kw.param("LOAD CASE") {
        let params = vec![("LOAD CASE".to_string(), lc.to_string()), ("BLOCK".to_string(), (step.cloads.len() - 1).to_string())];
        step.extra.push(RawKeyword { name: "_LOAD CASE CLOAD".into(), params, file: kw.file.clone(), line: kw.line, ..Default::default() });
    }
    Ok(())
}

fn dload(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut block = DloadBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        amplitude: kw.param("AMPLITUDE").map(|s| s.to_uppercase()),
        items: Vec::new(),
        source: kw.location(),
    };
    if kw.has("FOLLOWER") || kw.has("CONSTANT RESULTANT") {
        return kw.err("FOLLOWER/CONSTANT RESULTANT is not supported yet");
    }
    for d in &kw.data {
        // user loads PnNU (DLOAD) may lack a magnitude
        let user = d.fields.get(1).is_some_and(|k| crate::user::loads::parse_nu(&k.trim().to_uppercase()).is_some());
        if d.fields.len() < 3 && !(user && d.fields.len() == 2) {
            return kw.err_at(d.line, "write 'element/elset, load type, magnitude[, direction ...]'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let mut kind = d.fields[1].trim().to_uppercase();
        let magnitude = match d.fields.get(2) {
            Some(f) => num(kw, d.line, f)?,
            None => 0.0,
        };
        if user {
            // DLOAD is selected by the element set name (user/loads.rs)
            let set = if crate::inp::parse_label(&d.fields[0]).is_some() { String::new() } else { d.fields[0].trim().to_uppercase() };
            kind = format!("{kind}@{set}");
        }
        let mut extra = Vec::new();
        for s in d.fields.iter().skip(3) {
            extra.push(num(kw, d.line, s)?);
        }
        for e in elems {
            block.items.push(DistLoad { elem: e, kind: kind.clone(), magnitude, extra: extra.clone() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}
