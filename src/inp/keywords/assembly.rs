//! *PART, *END PART, *ASSEMBLY, *END ASSEMBLY, *INSTANCE and *END INSTANCE.
//!
//! A part is read as a separate `FeModel`. *INSTANCE copies it into
//! the model with a translation (first data line) and a rotation about an axis
//! through two points (second data line: a, b, angle in degrees). The translation
//! is applied before the rotation.

use crate::fem::model::{FeModel, Placement, RawKeyword};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ASSEMBLY", kind: Kind::Model, handler: assembly },
    Keyword { name: "END ASSEMBLY", kind: Kind::Model, handler: end_assembly },
    Keyword { name: "END INSTANCE", kind: Kind::Model, handler: end_instance },
    Keyword { name: "END PART", kind: Kind::Model, handler: end_part },
    Keyword { name: "INSTANCE", kind: Kind::Model, handler: instance },
    Keyword { name: "PART", kind: Kind::Model, handler: part },
];

fn part(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() || ctx.in_assembly {
        return kw.err("*PART may not appear inside another part or in *ASSEMBLY");
    }
    let name = required(kw, "NAME")?.to_uppercase();
    let mut m = FeModel::new();
    m.instance(&name);
    ctx.part = Some((name, m));
    Ok(())
}

fn end_part(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    match ctx.part.take() {
        Some((name, m)) => {
            ctx.parts.insert(name, m);
            Ok(())
        }
        None => kw.err("*END PART without *PART"),
    }
}

fn assembly(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() {
        return kw.err("*ASSEMBLY inside a part");
    }
    if !ctx.model.nodes.is_empty() && !ctx.assembly {
        return kw.err("nodes outside *PART before *ASSEMBLY are not supported");
    }
    ctx.assembly = true;
    ctx.in_assembly = true;
    Ok(())
}

fn end_assembly(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.in_assembly {
        return kw.err("*END ASSEMBLY without *ASSEMBLY");
    }
    ctx.in_assembly = false;
    Ok(())
}

fn instance(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.in_assembly {
        return kw.err("*INSTANCE may only appear in *ASSEMBLY");
    }
    let name = required(kw, "NAME")?.to_uppercase();
    let pname = required(kw, "PART")?.to_uppercase();
    let Some(part) = ctx.parts.get(&pname) else {
        return kw.err(format!("part {pname} does not exist"));
    };
    let mut place = Placement { translation: [0.0; 3], rotation: None };
    if let Some(d) = kw.data.first() {
        let v = kw.floats(d)?;
        place.translation = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
    }
    if let Some(d) = kw.data.get(1) {
        let v = kw.floats(d)?;
        if v.len() < 7 {
            return kw.err_at(d.line, "the rotation must be 'ax, ay, az, bx, by, bz, angle'");
        }
        place.rotation = Some(([v[0], v[1], v[2]], [v[3], v[4], v[5]], v[6]));
    }
    if ctx.model.instance_index(&name).is_some() {
        return kw.err(format!("instance {name} already exists"));
    }
    let part = part.clone();
    let n_extra = ctx.model.extra.len();
    ctx.model.instantiate(&part, &name, place).or_else(|e| kw.err(e.0))?;
    // the part's *RIGID BODY and analytical surfaces apply to this instance
    for k in ctx.model.extra[n_extra..].iter_mut().filter(|k| k.has("_PART")) {
        k.params.push(("_INSTANCE".into(), name.clone()));
        k.params.push(("_PLACE".into(), crate::interactions::analytical::encode_placement(&place)));
    }
    ctx.instance = Some(name);
    Ok(())
}

fn end_instance(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.instance.take().is_none() {
        return kw.err("*END INSTANCE without *INSTANCE");
    }
    Ok(())
}
