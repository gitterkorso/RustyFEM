//! Matrisstrukturen för kavitetsstrålning (strålningskavitet, inte *FLUID CAVITY i `cavity.rs`) (*CAVITY DEFINITION): alla noder
//! på kavitetens ytor kopplas till varandra (synfaktormatrisen är full).
//! Flödena räknas av värmeledningen i `steps/heat_cavity.rs`.

use super::{Interaction, InteractionFactory};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CAVITY DEFINITION"], build: links }];

struct CavityLink {
    nodes: Vec<usize>,
}

impl Interaction for CavityLink {
    fn name(&self) -> String {
        "kavitetsstrålning".into()
    }
    fn connectivity(&self) -> Vec<Vec<usize>> {
        vec![self.nodes.clone()]
    }
}

fn links(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let mut nodes = Vec::new();
    for d in kw.records() {
        for (k, f) in d.fields.iter().enumerate() {
            // med SET PROPERTY är varannan post ytegenskapens namn
            if kw.has("SET PROPERTY") && k % 2 == 1 {
                continue;
            }
            let Some(s) = m.surfaces.get(&f.trim().to_uppercase()) else { continue };
            for (e, label) in &s.faces {
                let el = &m.elements[*e];
                let faces = el.etype.faces();
                let Some(fi) = label.get(1..).and_then(|x| x.parse::<usize>().ok()).filter(|&i| i >= 1 && i <= faces.len()) else { continue };
                nodes.extend(faces[fi - 1].iter().map(|&i| el.nodes[i]));
            }
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![Box::new(CavityLink { nodes })])
}
