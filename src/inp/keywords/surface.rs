//! *SURFACE (TYPE=ELEMENT or NODE). The surfaces are stored in `FeModel::surfaces`
//! and used by contact, tie/coupling and surface loads (*DSLOAD).
//!
//! Data lines: `element/elset, face identifier` (e.g. S3, SPOS) for
//! element faces, `node/nset[, area]` for node-based surfaces. A line with only an elset
//! and no identifier gives the elements' free faces – that is not supported yet.

use crate::fem::model::{parse_f64, RawKeyword, Surface};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "SURFACE", kind: Kind::Model, handler: surface }];

fn surface(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "NAME")?.to_uppercase();
    let kind = kw.param_upper("TYPE").unwrap_or_else(|| "ELEMENT".into());
    // analytical rigid surfaces: inp/keywords/analytical.rs
    if crate::inp::keywords::analytical::is_analytical_type(&kind) {
        return crate::inp::keywords::analytical::surface(ctx, kw, &name);
    }
    let mut sf = Surface { kind: kind.clone(), source: kw.location(), ..Default::default() };
    for d in &kw.data {
        let f: Vec<&str> = d.fields.iter().map(|s| s.as_str()).filter(|s| !s.is_empty()).collect();
        let Some(first) = f.first() else { continue };
        match kind.as_str() {
            "ELEMENT" => {
                let Some(face) = f.get(1) else {
                    // Area explicit_adv: particle elements (PD3D, PC3D) have no face identifier
                    let es = ctx.elem_ref(kw, d.line, first, None)?;
                    if !es.is_empty() && es.iter().all(|&e| crate::elements::explicit_adv::is_particle(ctx.mesh().elements[e].etype.name())) {
                        sf.faces.extend(es.into_iter().map(|e| (e, "PARTICLE".to_string())));
                        continue;
                    }
                    return kw.err_at(d.line, "face identifier is missing (e.g. S1) – free faces are not supported yet");
                };
                let face = face.trim().to_uppercase();
                for e in ctx.elem_ref(kw, d.line, first, None)? {
                    sf.faces.push((e, face.clone()));
                }
            }
            "NODE" => {
                let w = match f.get(1) {
                    Some(s) => parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{} line {}: '{s}' is not a number", kw.file, d.line)))?,
                    None => 1.0,
                };
                for n in ctx.node_ref(kw, d.line, first, None)? {
                    sf.nodes.push((n, w));
                }
            }
            other => return kw.err(format!("TYPE={other} is not supported yet (ELEMENT, NODE)")),
        }
    }
    if ctx.mesh().surfaces.insert(name.clone(), sf).is_some() {
        return kw.err(format!("surface {name} is defined twice"));
    }
    Ok(())
}
