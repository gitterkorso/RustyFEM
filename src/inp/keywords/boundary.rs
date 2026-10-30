//! *BOUNDARY i modelldata och i steg.
//!
//! Dataradsformer: `nod/nset, första dof[, sista dof[, värde]]` eller
//! `nod/nset, TYP` där TYP är ENCASTRE, PINNED, XSYMM, YSYMM, ZSYMM, XASYMM,
//! YASYMM eller ZASYMM.

use crate::fem::dof::{self, Dof};
use crate::fem::model::{parse_f64, BcBlock, RawKeyword};
use crate::inp::{parse_label, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "BOUNDARY", kind: Kind::Both, handler: boundary }];

fn named(t: &str) -> Option<&'static [Dof]> {
    Some(match t {
        "ENCASTRE" => &[1, 2, 3, 4, 5, 6],
        "PINNED" => &[1, 2, 3],
        "XSYMM" => &[1, 5, 6],
        "YSYMM" => &[2, 4, 6],
        "ZSYMM" => &[3, 4, 5],
        "XASYMM" => &[2, 3, 4],
        "YASYMM" => &[1, 3, 5],
        "ZASYMM" => &[1, 2, 6],
        // område beams3: rörkrökarnas tvärsnitt (ersätts av elements::elbow::setup)
        "NOWARP" => &[crate::elements::elbow::NOWARP],
        "NOOVAL" => &[crate::elements::elbow::NOOVAL],
        "NODEFORM" => &[crate::elements::elbow::NODEFORM],
        _ => return None,
    })
}

fn boundary(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut block = BcBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        kind: kw.param_upper("TYPE").unwrap_or_else(|| "DISPLACEMENT".into()),
        amplitude: kw.param("AMPLITUDE").map(|s| s.to_uppercase()),
        items: Vec::new(),
        source: kw.location(),
    };
    if kw.has("SUBMODEL") {
        // submodellering (steps/submodel.rs)
        return crate::inp::keywords::infrastructure::boundary_submodel(ctx, kw);
    }
    if kw.has("FIXED") {
        return kw.err("FIXED stöds inte ännu");
    }
    // *BOUNDARY, USER: DISP ger värdena (user/loads.rs)
    let mut user = crate::fem::model::RawKeyword { name: crate::user::loads::USER_BC.into(), file: kw.file.clone(), line: kw.line, ..Default::default() };
    for d in &kw.data {
        let f: Vec<&str> = d.fields.iter().map(|s| s.as_str()).collect();
        if f.len() < 2 {
            return kw.err_at(d.line, "skriv 'nod, första dof[, sista dof[, värde]]' eller 'nod, ENCASTRE'");
        }
        let nodes = ctx.node_ref(kw, d.line, f[0], None)?;
        let spec = f[1].trim().to_uppercase();
        let (dofs, value): (Vec<Dof>, f64) = if let Some(list) = named(&spec) {
            (list.to_vec(), 0.0)
        } else {
            let Some(first) = parse_label(&spec) else { return kw.err_at(d.line, format!("'{}' är ingen frihetsgrad", f[1])) };
            let last = match f.get(2).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                Some(s) => parse_label(s).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{s}' är ingen frihetsgrad", kw.file, d.line)))?,
                None => first,
            };
            let value = match f.get(3) {
                Some(s) => parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{s}' är inget tal", kw.file, d.line)))?,
                None => 0.0,
            };
            if first < 1 || last < first {
                return kw.err_at(d.line, "ogiltigt frihetsgradsintervall");
            }
            let mut v = Vec::new();
            for k in first..=last {
                let dd = k as Dof;
                if dof::slot(dd).is_none() {
                    return kw.err_at(d.line, format!("frihetsgrad {k} stöds inte (1–7 och 11)"));
                }
                v.push(dd);
            }
            (v, value)
        };
        for &n in &nodes {
            for &dd in &dofs {
                block.items.push((n, dd, value));
                if kw.has("USER") {
                    user.data.push(crate::fem::model::DataLine { fields: vec![n.to_string(), dd.to_string(), format!("{value:e}")], line: d.line, continued: false });
                }
            }
        }
        if kw.has("USER") && parse_label(f[0]).is_none() && user.param("_NAME").is_none() {
            user.params.push(("_NAME".into(), f[0].trim().to_uppercase()));
        }
    }
    if kw.has("USER") {
        if !user.has("_NAME") {
            user.params.push(("_NAME".into(), String::new()));
        }
        match &mut ctx.step {
            Some(s) => s.extra.push(user),
            None => return kw.err("*BOUNDARY, USER stöds bara i steg"),
        }
    }
    match &mut ctx.step {
        Some(s) => s.bcs.push(block),
        None => {
            if ctx.part.is_some() {
                return kw.err("*BOUNDARY i en *PART stöds inte – lägg den efter *END ASSEMBLY eller i ett steg");
            }
            ctx.model.initial_bcs.push(block)
        }
    }
    Ok(())
}
