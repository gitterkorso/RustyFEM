//! *PART, *END PART, *ASSEMBLY, *END ASSEMBLY, *INSTANCE och *END INSTANCE.
//!
//! En part läses in som en egen `FeModel`. *INSTANCE kopierar den till
//! modellen med förflyttning (första dataraden) och rotation kring en axel
//! genom två punkter (andra dataraden: a, b, vinkel i grader). Förflyttningen
//! görs före rotationen.

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
        return kw.err("*PART får inte stå inne i en annan part eller i *ASSEMBLY");
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
        None => kw.err("*END PART utan *PART"),
    }
}

fn assembly(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() {
        return kw.err("*ASSEMBLY inne i en part");
    }
    if !ctx.model.nodes.is_empty() && !ctx.assembly {
        return kw.err("noder utanför *PART före *ASSEMBLY stöds inte");
    }
    ctx.assembly = true;
    ctx.in_assembly = true;
    Ok(())
}

fn end_assembly(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.in_assembly {
        return kw.err("*END ASSEMBLY utan *ASSEMBLY");
    }
    ctx.in_assembly = false;
    Ok(())
}

fn instance(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.in_assembly {
        return kw.err("*INSTANCE får bara stå i *ASSEMBLY");
    }
    let name = required(kw, "NAME")?.to_uppercase();
    let pname = required(kw, "PART")?.to_uppercase();
    let Some(part) = ctx.parts.get(&pname) else {
        return kw.err(format!("parten {pname} finns inte"));
    };
    let mut place = Placement { translation: [0.0; 3], rotation: None };
    if let Some(d) = kw.data.first() {
        let v = kw.floats(d)?;
        place.translation = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
    }
    if let Some(d) = kw.data.get(1) {
        let v = kw.floats(d)?;
        if v.len() < 7 {
            return kw.err_at(d.line, "rotationen ska vara 'ax, ay, az, bx, by, bz, vinkel'");
        }
        place.rotation = Some(([v[0], v[1], v[2]], [v[3], v[4], v[5]], v[6]));
    }
    if ctx.model.instance_index(&name).is_some() {
        return kw.err(format!("instansen {name} finns redan"));
    }
    let part = part.clone();
    let n_extra = ctx.model.extra.len();
    ctx.model.instantiate(&part, &name, place).or_else(|e| kw.err(e.0))?;
    // partens *RIGID BODY och analytiska ytor gäller den här instansen
    for k in ctx.model.extra[n_extra..].iter_mut().filter(|k| k.has("_PART")) {
        k.params.push(("_INSTANCE".into(), name.clone()));
        k.params.push(("_PLACE".into(), crate::interactions::analytical::encode_placement(&place)));
    }
    ctx.instance = Some(name);
    Ok(())
}

fn end_instance(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.instance.take().is_none() {
        return kw.err("*END INSTANCE utan *INSTANCE");
    }
    Ok(())
}
