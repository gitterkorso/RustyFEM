//! *NSET and *ELSET (with GENERATE, INSTANCE= and set names on the data lines).

use crate::fem::model::RawKeyword;
use crate::inp::{parse_label, required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ELSET", kind: Kind::Model, handler: elset },
    Keyword { name: "NSET", kind: Kind::Model, handler: nset },
];

/// All entries on the data lines: labels (or ranges with GENERATE) and set names.
fn members(ctx: &InpCtx, kw: &RawKeyword, nodes: bool) -> Result<Vec<usize>> {
    let inst = kw.param("INSTANCE").map(|s| s.trim().to_uppercase());
    let generate = kw.has("GENERATE");
    let mut out = Vec::new();
    for d in &kw.data {
        let toks: Vec<&str> = d.fields.iter().map(|s| s.as_str()).filter(|s| !s.is_empty()).collect();
        if generate {
            let v: Vec<i64> = toks.iter().filter_map(|t| parse_label(t)).collect();
            if v.len() < 2 || v.len() != toks.len() {
                return kw.err_at(d.line, "GENERATE requires 'first, last[, increment]'");
            }
            let step = v.get(2).copied().unwrap_or(1).max(1);
            let mut l = v[0];
            while l <= v[1] {
                let t = l.to_string();
                out.extend(if nodes { ctx.node_ref(kw, d.line, &t, inst.as_deref())? } else { ctx.elem_ref(kw, d.line, &t, inst.as_deref())? });
                l += step;
            }
        } else {
            for t in toks {
                out.extend(if nodes { ctx.node_ref(kw, d.line, t, inst.as_deref())? } else { ctx.elem_ref(kw, d.line, t, inst.as_deref())? });
            }
        }
    }
    Ok(out)
}

fn nset(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "NSET")?.to_string();
    let mut ids = Vec::new();
    if let Some(es) = kw.param("ELSET") {
        let elems = ctx.elem_ref(kw, kw.line, es, kw.param("INSTANCE"))?;
        let mesh = ctx.mesh_ref();
        for e in elems {
            ids.extend(mesh.elements[e].nodes.iter().copied());
        }
    }
    ids.extend(members(ctx, kw, true)?);
    ctx.mesh().add_to_nset(&name, &ids);
    Ok(())
}

fn elset(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "ELSET")?.to_string();
    let ids = members(ctx, kw, false)?;
    ctx.mesh().add_to_elset(&name, &ids);
    Ok(())
}
