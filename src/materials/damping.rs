//! *DAMPING (Rayleigh: ALPHA, BETA) läses av dynamikstegen (`steps/dynamic.rs`)
//! och förbrukas därför passivt här.

use super::MaterialFactory;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory { name: "Rayleigh-dämpning (dynamiksteg)", consumes: &["DAMPING"], priority: 0, build: None }];
