//! *RIGID BODY (rigid bodies). The keyword is stored in the model and parsed by
//! `interactions/rigid_body.rs`. Rigid elements (R3D4, R3D3, R2D2, RAX2)
//! are in `elements/rigid.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "RIGID BODY", kind: Kind::Model, handler: rigid_body }];

fn rigid_body(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    crate::inp::required(kw, "REF NODE")?;
    let mut k = kw.clone();
    // inside a part the names refer to the part's nodes and sets; *INSTANCE adds
    // the instance name (see assembly.rs and interactions/rigid_body.rs)
    if let Some((name, _)) = &ctx.part {
        k.params.push(("_PART".into(), name.clone()));
    }
    ctx.mesh().extra.push(k);
    Ok(())
}
