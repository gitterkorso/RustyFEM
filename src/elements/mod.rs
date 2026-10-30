//! Element library with a registry by element name.
//!
//! One file per element family. Each module exports
//! `pub static TYPES: &[&dyn ElementType]`. Register a new module with ONE line
//! in the list below (alphabetical order). The interface is in `api.rs`,
//! shared building blocks in `common.rs`.

mod api;
pub mod common;
pub use api::*;

use std::collections::HashMap;
use std::sync::OnceLock;

macro_rules! element_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered element types.
        pub fn all() -> Vec<&'static dyn ElementType> {
            let mut v: Vec<&'static dyn ElementType> = Vec::new();
            $(v.extend($m::TYPES.iter().copied());)*
            v
        }
    };
}

// One line per element family, in alphabetical order.
element_modules! {
    acoustic,
    acoustic_explicit,
    acoustic_infinite,
    axisym,
    beam_eb,
    beam_eb_curved,
    beam_mass,
    beam_timoshenko,
    beam_twist,
    beams2,
    cohesive,
    continuum2d,
    continuum3d,
    elbow,
    electromagnetic,
    elements3,
    explicit_adv,
    gasket,
    geometric,
    heat,
    heat_convection,
    heat_coupled,
    heat_coupled_reduced,
    hybrid,
    hybrid2,
    hybrid_tet,
    infinite,
    matrix_element,
    normals,
    pipe,
    piezo,
    point,
    rigid,
    shell,
    shell2,
    shell3,
    solid_higher,
    solid_higher_gps,
    solid_modified,
    soils,
    thermal_electric,
    thermal_electric_structural,
    truss,
    user_element,
    xfem,
}

fn table() -> &'static HashMap<&'static str, &'static dyn ElementType> {
    static T: OnceLock<HashMap<&'static str, &'static dyn ElementType>> = OnceLock::new();
    T.get_or_init(|| {
        let mut m = HashMap::new();
        for t in all() {
            if m.insert(t.name(), t).is_some() {
                panic!("element type {} is registered twice", t.name());
            }
        }
        m
    })
}

/// Element type for an element name (upper case), e.g. "CPS4".
pub fn lookup(name: &str) -> Option<&'static dyn ElementType> {
    table().get(name.to_uppercase().as_str()).copied()
}

/// Names of all registered element types, sorted.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&str> = table().keys().copied().collect();
    v.sort_unstable();
    v
}
