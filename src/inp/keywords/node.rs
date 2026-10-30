//! *NODE

use crate::fem::model::RawKeyword;
use crate::inp::{parse_label, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "NODE", kind: Kind::Model, handler: node }];

fn node(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if let Some(sys) = kw.param_upper("SYSTEM") {
        if sys != "R" {
            return kw.err(format!("SYSTEM={sys} (cylindriska/sfäriska koordinater) stöds inte ännu"));
        }
    }
    if ctx.instance.is_some() {
        return kw.err("noder inne i *INSTANCE stöds inte – definiera dem i en *PART");
    }
    let inst = ctx.mesh_instance();
    let mut ids = Vec::with_capacity(kw.data.len());
    for d in &kw.data {
        let Some(label) = d.fields.first().and_then(|f| parse_label(f)) else {
            return kw.err_at(d.line, "nodnummer saknas");
        };
        let v = kw.floats(&crate::fem::model::DataLine { fields: d.fields[1..].to_vec(), line: d.line, continued: false })?;
        let x = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
        ids.push(ctx.mesh().add_node(inst, label, x));
    }
    if let Some(set) = kw.param("NSET") {
        ctx.mesh().add_to_nset(set, &ids);
    }
    Ok(())
}
