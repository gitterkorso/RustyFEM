//! Keyword handlers, one file per area.
//!
//! Each module exports `pub static KEYWORDS: &[Keyword]`. Register a new
//! module with ONE line in the list below (alphabetical order). A keyword may
//! only be registered once (checked by a test).

use super::Keyword;

macro_rules! keyword_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered keywords.
        pub fn all() -> Vec<&'static Keyword> {
            let mut v = Vec::new();
            $(v.extend($m::KEYWORDS.iter());)*
            v
        }
    };
}

// One line per module, in alphabetical order.
keyword_modules! {
    acoustic,
    acoustics2,
    analytical,
    assembly,
    boundary,
    cavity,
    cohesive,
    cohesive_contact,
    connectors,
    contact3,
    coupled2,
    dsload,
    dyn_incr,
    dynamics,
    eigen,
    element,
    eos,
    explicit,
    explicit_adv,
    foam,
    gaps1,
    geomaterials,
    fracture,
    gasket,
    heat,
    infrastructure,
    interactions,
    loads,
    material,
    material_behaviors,
    materials_soil,
    materials_time,
    materials3,
    materials4,
    materials5,
    metal_plasticity,
    modal,
    multiphysics,
    nlgeom,
    node,
    normals,
    output,
    physics3,
    procedures,
    procedures2,
    rigid,
    section,
    sets,
    shells_beams,
    soils,
    step,
    surface,
    temperature,
    user,
    user2,
    xfem,
}
