//! *RIGID BODY (stela kroppar). Nyckelordet sparas i modellen och tolkas av
//! `interactions/rigid_body.rs`. Stela element (R3D4, R3D3, R2D2, RAX2)
//! finns i `elements/rigid.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "RIGID BODY", kind: Kind::Model, handler: rigid_body }];

fn rigid_body(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    crate::inp::required(kw, "REF NODE")?;
    let mut k = kw.clone();
    // inne i en part syftar namnen på partens noder och set; *INSTANCE lägger
    // till instansens namn (se assembly.rs och interactions/rigid_body.rs)
    if let Some((name, _)) = &ctx.part {
        k.params.push(("_PART".into(), name.clone()));
    }
    ctx.mesh().extra.push(k);
    Ok(())
}
