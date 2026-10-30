//! Area acoustics2: *INITIAL CONDITIONS, TYPE=ACOUSTIC STATIC PRESSURE
//! (called from `temperature.rs`). The static pressure is only used in
//! the cavitation condition in explicit analysis (`steps/acoustic_explicit.rs`).
//!
//! Data line: node or node set, p₁, x₁, y₁, z₁[, p₂, x₂, y₂, z₂]. With two
//! reference points the pressure varies linearly along the line between them
//! (the projection of the node position), otherwise it is p₁.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[];

/// Parameter that marks resolved lines (node index, p₀) in `FeModel::extra`.
pub const RESOLVED: &str = "RUSTYFEM_ACOUSTIC_STATIC";

pub fn initial_static_pressure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut out = kw.clone();
    out.params.push((RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = d.fields[1..].iter().map(|f| crate::fem::model::parse_f64(f.trim()).unwrap_or(0.0)).collect();
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let (p1, x1) = (g(0), [g(1), g(2), g(3)]);
        let two = v.len() > 4;
        let (p2, x2) = (g(4), [g(5), g(6), g(7)]);
        let d12 = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]];
        let l2 = d12[0] * d12[0] + d12[1] * d12[1] + d12[2] * d12[2];
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            let x = ctx.model.nodes[n];
            let p = if two && l2 > 0.0 {
                let s = ((x[0] - x1[0]) * d12[0] + (x[1] - x1[1]) * d12[1] + (x[2] - x1[2]) * d12[2]) / l2;
                p1 + s * (p2 - p1)
            } else {
                p1
            };
            out.data.push(DataLine { fields: vec![n.to_string(), format!("{p:e}")], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
