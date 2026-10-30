//! Material models and their registry.
//!
//! Each module exports `pub static FACTORIES: &[MaterialFactory]`.
//! Register a new module with ONE line in the list below (alphabetical order).

mod api;
pub use api::*;

use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

macro_rules! material_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered material factories.
        pub fn factories() -> Vec<&'static MaterialFactory> {
            let mut v = Vec::new();
            $(v.extend($m::FACTORIES.iter());)*
            v
        }
    };
}

// One line per module, in alphabetical order.
material_modules! {
    acoustic,
    anisotropic,
    cap,
    cohesive,
    clay,
    concrete_damaged,
    concrete_smeared,
    creep,
    creep_ext,
    crushable_foam,
    damage_tables,
    damping,
    deformation_plasticity,
    ductile_damage,
    elastic,
    electromagnetic,
    gasket,
    eos,
    field_dependent,
    field_vars,
    forming_limit,
    hashin,
    heat,
    hyperelastic,
    hyperelastic_fit,
    hyperelastic_ogden,
    hyperelastic_split,
    hyperelastic_testdata,
    hyperelastic_vdw,
    hyperfoam,
    incompressible_elastic,
    inelastic_heat,
    initial_state,
    invariant_return,
    mohr_coulomb,
    mullins,
    multiphysics,
    plasticity,
    plasticity_dp,
    plasticity_dp2,
    plasticity_hill,
    plasticity_rate,
    plasticity_rate_tables,
    porous,
    porous_metal,
    principal,
    thermal,
    thermal_elastic,
    time_dependence,
    user_material,
    user_routines,
    user_vumat,
    viscoelastic,
    viscoelastic_freq,
    viscoelastic_trs,
    xfem_damage,
}

/// Behaviors that the core handles itself (always consumed).
const CORE_BEHAVIORS: &[&str] = &["DENSITY"];

/// Builds the material model for a material definition.
pub fn build(def: &MaterialDef) -> Result<Arc<dyn MaterialModel>> {
    // *ELASTIC with ν = 0.5 (hybrid elements): rebuilt with ν = 0.4999999
    if let Some(m) = incompressible_elastic::claim(def)? {
        return Ok(m);
    }
    // field-dependent data (DEPENDENCIES=) and USDFLD are resolved in field_dependent.rs
    if let Some(m) = field_dependent::claim(def)? {
        return Ok(m);
    }
    // UHARD (*PLASTIC, HARDENING=USER) and UEXPAN (*EXPANSION, USER)
    if let Some(m) = user_routines::claim(def)? {
        return Ok(m);
    }
    let all = factories();
    let mut best: Option<(i32, &MaterialFactory, Arc<dyn MaterialModel>)> = None;
    for f in &all {
        let Some(b) = f.build else { continue };
        if let Some(m) = b(def)? {
            if best.as_ref().map_or(true, |(p, _, _)| f.priority > *p) {
                best = Some((f.priority, f, m));
            }
        }
    }
    let Some((_, fac, model)) = best else {
        let names: Vec<&str> = def.behaviors.iter().map(|b| b.name.as_str()).collect();
        return fail(format!(
            "material {}: no material model in RustyFEM handles the behaviors [{}] – see src/materials/",
            def.name,
            names.join(", ")
        ));
    };
    for b in &def.behaviors {
        let used = fac.consumes.contains(&b.name.as_str())
            || CORE_BEHAVIORS.contains(&b.name.as_str())
            || all.iter().any(|f| f.build.is_none() && f.consumes.contains(&b.name.as_str()));
        if !used {
            return fail(format!(
                "material {}: behavior *{} is not supported by material model {} ({})",
                def.name,
                b.name,
                fac.name,
                b.location()
            ));
        }
    }
    Ok(model)
}
