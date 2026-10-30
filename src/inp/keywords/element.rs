//! *ELEMENT

use crate::elements;
use crate::fem::model::RawKeyword;
use crate::inp::{parse_label, required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "ELEMENT", kind: Kind::Model, handler: element }];

fn element(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = required(kw, "TYPE")?.to_uppercase();
    // användarelement (TYPE=Un) definieras av *USER ELEMENT i filen
    let etype = match elements::lookup(&ty) {
        Some(t) => Some(t),
        None => match crate::elements::user_element::lookup(ctx, &ty)? {
            Some(t) => Some(t),
            // substrukturer (TYPE=Zn, FILE=jobb) från *SUBSTRUCTURE GENERATE
            None => crate::elements::matrix_element::lookup_substructure(ctx, kw, &ty)?,
        },
    };
    let Some(etype) = etype else {
        return kw.err(format!("elementtypen {ty} finns inte i RustyFEM (registrerade: {})", elements::names().join(", ")));
    };
    if ctx.instance.is_some() {
        return kw.err("element inne i *INSTANCE stöds inte – definiera dem i en *PART");
    }
    let inst = ctx.mesh_instance();
    let nn = etype.n_connectivity();
    let mut ids = Vec::new();
    for rec in kw.records() {
        let f: Vec<&String> = rec.fields.iter().filter(|s| !s.is_empty()).collect();
        let Some(label) = f.first().and_then(|s| parse_label(s)) else {
            return kw.err_at(rec.line, "elementnummer saknas");
        };
        if f.len() - 1 != nn {
            return kw.err_at(rec.line, format!("{ty} ska ha {nn} noder, raden har {}", f.len() - 1));
        }
        let mut nodes = Vec::with_capacity(nn);
        for s in &f[1..] {
            let Some(l) = parse_label(s) else { return kw.err_at(rec.line, format!("'{s}' är inget nodnummer")) };
            match ctx.mesh_ref().node(inst, l) {
                Some(n) => nodes.push(n),
                None => return kw.err_at(rec.line, format!("element {label}: nod {l} finns inte")),
            }
        }
        let id = ctx.mesh().add_element(etype, inst, label, nodes).or_else(|e| kw.err_at(rec.line, e.0))?;
        ids.push(id);
    }
    if let Some(set) = kw.param("ELSET") {
        ctx.mesh().add_to_elset(set, &ids);
    }
    // område elements3: SAXA-elementens plannoder (nod + 100000·k) skapas direkt
    // så att *BOUNDARY och *CLOAD kan ange dem
    crate::elements::elements3::saxa::create_plane_nodes(ctx.mesh(), &ids);
    Ok(())
}
