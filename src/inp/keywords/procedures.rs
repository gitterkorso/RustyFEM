//! Stegprocedurfunktioner: *MODEL CHANGE (borttagning och återaktivering av
//! element och kontaktpar, tolkas av `steps/model_change.rs`).
//!
//! *STATIC, STABILIZE läses av `steps/static_stabilize.rs` direkt ur
//! procedurnyckelordet.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "MODEL CHANGE", kind: Kind::Step, handler: model_change }];

/// Sparar *MODEL CHANGE i steget. För TYPE=ELEMENT ersätts dataraderna av
/// elementens index ("#index") så att set i instanser löses upp här.
fn model_change(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ELEMENT".into());
    if kw.has("ACTIVATE") && !kw.has("REMOVE") && !kw.has("ADD") {
        // förbereder bara omstart: ingen ändring
        return Ok(());
    }
    let mut stored = kw.clone();
    if ty == "ELEMENT" {
        let mut idx: Vec<usize> = Vec::new();
        for d in kw.records() {
            for f in d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                idx.extend(ctx.elem_ref(kw, d.line, f, None)?);
            }
        }
        idx.sort_unstable();
        idx.dedup();
        stored.data = vec![DataLine { fields: idx.iter().map(|e| format!("#{e}")).collect(), line: kw.line, continued: false }];
    }
    ctx.step_mut(kw)?.extra.push(stored);
    Ok(())
}
