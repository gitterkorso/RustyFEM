//! Step procedure features: *MODEL CHANGE (removal and reactivation of
//! elements and contact pairs, parsed by `steps/model_change.rs`).
//!
//! *STATIC, STABILIZE is read by `steps/static_stabilize.rs` directly from
//! the procedure keyword.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "MODEL CHANGE", kind: Kind::Step, handler: model_change }];

/// Stores *MODEL CHANGE in the step. For TYPE=ELEMENT the data lines are replaced by
/// the element indices ("#index") so that sets in instances are resolved here.
fn model_change(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ELEMENT".into());
    if kw.has("ACTIVATE") && !kw.has("REMOVE") && !kw.has("ADD") {
        // only prepares for restart: no change
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
