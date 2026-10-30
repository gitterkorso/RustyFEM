//! *SURFACE (TYPE=ELEMENT eller NODE). Ytorna lagras i `FeModel::surfaces`
//! och används av kontakt, tie/coupling och ytlaster (*DSLOAD).
//!
//! Dataraderna: `element/elset, ytbeteckning` (t.ex. S3, SPOS) för
//! elementytor, `nod/nset[, area]` för nodytor. En rad med bara ett elset
//! utan beteckning ger elementens fria ytor – det stöds inte ännu.

use crate::fem::model::{parse_f64, RawKeyword, Surface};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "SURFACE", kind: Kind::Model, handler: surface }];

fn surface(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "NAME")?.to_uppercase();
    let kind = kw.param_upper("TYPE").unwrap_or_else(|| "ELEMENT".into());
    // analytiska stela ytor: inp/keywords/analytical.rs
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
                    // område explicit_adv: partikelelement (PD3D, PC3D) har ingen ytbeteckning
                    let es = ctx.elem_ref(kw, d.line, first, None)?;
                    if !es.is_empty() && es.iter().all(|&e| crate::elements::explicit_adv::is_particle(ctx.mesh().elements[e].etype.name())) {
                        sf.faces.extend(es.into_iter().map(|e| (e, "PARTICLE".to_string())));
                        continue;
                    }
                    return kw.err_at(d.line, "ytbeteckning saknas (t.ex. S1) – fria ytor stöds inte ännu");
                };
                let face = face.trim().to_uppercase();
                for e in ctx.elem_ref(kw, d.line, first, None)? {
                    sf.faces.push((e, face.clone()));
                }
            }
            "NODE" => {
                let w = match f.get(1) {
                    Some(s) => parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{s}' är inget tal", kw.file, d.line)))?,
                    None => 1.0,
                };
                for n in ctx.node_ref(kw, d.line, first, None)? {
                    sf.nodes.push((n, w));
                }
            }
            other => return kw.err(format!("TYPE={other} stöds inte ännu (ELEMENT, NODE)")),
        }
    }
    if ctx.mesh().surfaces.insert(name.clone(), sf).is_some() {
        return kw.err(format!("ytan {name} är definierad två gånger"));
    }
    Ok(())
}
