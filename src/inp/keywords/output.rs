//! Rubrik, utdata- och diagnostiknyckelord. RustyFEM skriver alltid sina
//! utdatafält; begäran sparas i steget (`StepDef::output`) för senare bruk.

use crate::fem::model::RawKeyword;
use crate::inp::{ignore, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONTACT OUTPUT", kind: Kind::Step, handler: request },
    Keyword { name: "CONTACT PRINT", kind: Kind::Step, handler: request },
    Keyword { name: "DIAGNOSTICS", kind: Kind::Both, handler: ignore },
    Keyword { name: "EL FILE", kind: Kind::Step, handler: request },
    Keyword { name: "EL PRINT", kind: Kind::Step, handler: request },
    Keyword { name: "ELEMENT OUTPUT", kind: Kind::Step, handler: request },
    Keyword { name: "ENERGY FILE", kind: Kind::Step, handler: request },
    Keyword { name: "ENERGY OUTPUT", kind: Kind::Step, handler: request },
    Keyword { name: "ENERGY PRINT", kind: Kind::Step, handler: request },
    Keyword { name: "FILE FORMAT", kind: Kind::Both, handler: ignore },
    Keyword { name: "HEADING", kind: Kind::Model, handler: heading },
    Keyword { name: "MONITOR", kind: Kind::Step, handler: request },
    Keyword { name: "NODE FILE", kind: Kind::Step, handler: request },
    Keyword { name: "NODE OUTPUT", kind: Kind::Step, handler: request },
    Keyword { name: "NODE PRINT", kind: Kind::Step, handler: request },
    Keyword { name: "OUTPUT", kind: Kind::Step, handler: request },
    Keyword { name: "PREPRINT", kind: Kind::Model, handler: ignore },
    Keyword { name: "PRINT", kind: Kind::Step, handler: request },
    Keyword { name: "SECTION PRINT", kind: Kind::Step, handler: request },
];

fn heading(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.model.heading = kw.data.iter().map(|d| d.fields.join(",")).collect::<Vec<_>>().join("\n");
    Ok(())
}

fn request(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    ctx.step_mut(kw)?.output.push(kw.clone());
    Ok(())
}
