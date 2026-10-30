//! *DAMPING (Rayleigh: ALPHA, BETA) is read by the dynamic steps (`steps/dynamic.rs`)
//! and is therefore consumed passively here.

use super::MaterialFactory;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory { name: "Rayleigh damping (dynamic steps)", consumes: &["DAMPING"], priority: 0, build: None }];
