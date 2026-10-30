//! Elementbibliotek med register per elementnamn.
//!
//! En fil per elementfamilj. Varje modul exporterar
//! `pub static TYPES: &[&dyn ElementType]`. Registrera en ny modul med EN rad
//! i listan nedan (alfabetisk ordning). Gränssnittet finns i `api.rs`,
//! gemensamma byggstenar i `common.rs`.

mod api;
pub mod common;
pub use api::*;

use std::collections::HashMap;
use std::sync::OnceLock;

macro_rules! element_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// Alla registrerade elementtyper.
        pub fn all() -> Vec<&'static dyn ElementType> {
            let mut v: Vec<&'static dyn ElementType> = Vec::new();
            $(v.extend($m::TYPES.iter().copied());)*
            v
        }
    };
}

// En rad per elementfamilj, i alfabetisk ordning.
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
                panic!("elementtypen {} är registrerad två gånger", t.name());
            }
        }
        m
    })
}

/// Elementtyp för ett elementnamn (versaler), t.ex. "CPS4".
pub fn lookup(name: &str) -> Option<&'static dyn ElementType> {
    table().get(name.to_uppercase().as_str()).copied()
}

/// Namnen på alla registrerade elementtyper, sorterade.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&str> = table().keys().copied().collect();
    v.sort_unstable();
    v
}
