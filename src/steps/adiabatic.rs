//! Adiabatic analysis (area coupled2): *STATIC, ADIABATIC and
//! *DYNAMIC, ADIABATIC (implicit analysis).
//!
//! The temperature is computed at the integration points without heat conduction:
//! ΔT = η σ:Δεᵖˡ / (ρc) per increment, with η from *INELASTIC HEAT FRACTION.
//! The material wrapper `materials::inelastic_heat::Adiabatic` does the heating;
//! the step type here switches on adiabatic mode at all integration points
//! (initial temperature = the element's nodal temperature), runs the step with the
//! regular step type for *STATIC or *DYNAMIC, and switches the mode off
//! afterwards (the temperature at the points is kept). TEMP in element output.

use super::api::{StepCtx, StepType};
use crate::fem::model::StepDef;
use crate::materials::inelastic_heat::{set_active, N_EXTRA, NAME};
use crate::model::{fail, Result};

pub struct Adiabatic {
    dynamic: bool,
}

static STATIC: Adiabatic = Adiabatic { dynamic: false };
static DYNAMIC: Adiabatic = Adiabatic { dynamic: true };

pub static STEP_TYPES: &[&dyn StepType] = &[&STATIC, &DYNAMIC];

fn proc_name(dynamic: bool) -> &'static str {
    if dynamic {
        "DYNAMIC"
    } else {
        "STATIC"
    }
}

/// Sets adiabatic mode at all integration points with an `Adiabatic` material.
fn set_mode(ctx: &mut StepCtx, on: bool) {
    let p = ctx.prep;
    let m = p.model;
    for (e, el) in m.elements.iter().enumerate() {
        let ectx = p.ctx(e, false, ctx.state.time, 0.0);
        let Some(mat) = ectx.material else { continue };
        if mat.name() != NAME {
            continue;
        }
        let nip = el.etype.n_ips();
        let nc = crate::materials::inelastic_heat::stress_components(el);
        let stride = nc + mat.n_state();
        let st = &mut ctx.state.elem_state[e];
        if nip == 0 || st.len() < nip * stride || mat.n_state() < N_EXTRA {
            continue;
        }
        // mean of the nodal temperatures (the initial temperature)
        let t0 = if ectx.temp.is_empty() { 0.0 } else { ectx.temp.iter().take(el.nodes.len()).sum::<f64>() / el.nodes.len() as f64 };
        for k in 0..nip {
            let b = k * stride + nc;
            set_active(mat, &mut st[b..b + mat.n_state()], on, t0);
        }
    }
}

impl StepType for Adiabatic {
    fn name(&self) -> &'static str {
        if self.dynamic {
            "DYNAMIC, ADIABATIC"
        } else {
            "STATIC, ADIABATIC"
        }
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (p.name == proc_name(self.dynamic) && p.has("ADIABATIC") && !p.has("EXPLICIT")).then_some(100)
    }

    fn uses_amplitudes(&self) -> bool {
        self.dynamic
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let model = ctx.prep.model;
        let me = self.name();
        let delegate = super::all().into_iter().filter(|t| t.name() != me).filter_map(|t| t.claim_model(ctx.step, ctx.nlgeom, model).map(|p| (p, t))).max_by_key(|(p, _)| *p);
        let Some((_, ty)) = delegate else {
            return fail(format!("*{}, ADIABATIC: no step type for the procedure", proc_name(self.dynamic)));
        };
        set_mode(ctx, true);
        let r = ty.run(ctx);
        set_mode(ctx, false);
        r
    }
}
