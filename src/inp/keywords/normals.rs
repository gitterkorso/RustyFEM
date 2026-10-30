//! *NORMAL (Area normals): the normal at a node for a given element (shell)
//! or the beam's n2 direction.
//!
//! The data lines "element (or element set), node, nx, ny[, nz]" are stored as
//! a keyword "NORMAL" in `FeModel::extra` with the lines
//! "instance, element, node, nx, ny, nz" (the instance empty and `_PART` in a part;
//! `*INSTANCE` then adds `_INSTANCE`/`_PLACE`). `elements/normals.rs`
//! uses them once the file has been read.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "NORMAL", kind: Kind::Model, handler: normal }];

fn normal(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if let Some(t) = kw.param_upper("TYPE") {
        if t != "ELEMENT" {
            return kw.err(format!("TYPE={t} is not supported (only TYPE=ELEMENT)"));
        }
    }
    let part = ctx.part.as_ref().map(|(n, _)| n.clone());
    let mut out = RawKeyword { name: "NORMAL".into(), file: kw.file.clone(), line: kw.line, ..Default::default() };
    if let Some(p) = &part {
        out.params.push(("_PART".into(), p.clone()));
    }
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "*NORMAL: the line must be element, node, nx, ny[, nz]");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let nodes = ctx.node_ref(kw, d.line, &d.fields[1], None)?;
        let v = kw.floats(&DataLine { fields: d.fields[2..].to_vec(), line: d.line, continued: false })?;
        let nv = [v[0], v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
        let mesh = ctx.mesh_ref();
        let before = out.data.len();
        for &n in &nodes {
            let nlabel = mesh.node_label[n];
            for &e in &elems {
                let el = &mesh.elements[e];
                if !el.nodes.contains(&n) {
                    continue;
                }
                let inst = if part.is_some() { String::new() } else { mesh.instances[el.inst as usize].clone() };
                let fields = vec![inst, el.label.to_string(), nlabel.to_string(), nv[0].to_string(), nv[1].to_string(), nv[2].to_string()];
                out.data.push(DataLine { fields, line: d.line, continued: false });
            }
        }
        if out.data.len() == before {
            return kw.err_at(d.line, "*NORMAL: the node does not belong to the element");
        }
    }
    ctx.mesh().extra.push(out);
    Ok(())
}
