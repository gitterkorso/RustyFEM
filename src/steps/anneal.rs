//! *ANNEAL: glödgning mellan två steg.
//!
//! Spänningarna nollställs och materialpunkten görs spänningsfri i den
//! aktuella deformerade geometrin (`MaterialModel::anneal`): för
//! von Mises-plasticitet blir den plastiska töjningen lika med den totala
//! töjningen (logaritmisk töjning vid NLGEOM, annars E) och PEEQ och mottensorerna noll.
//! Hastigheter, accelerationer och reaktioner nollställs; förskjutningen
//! behålls. Steget tar ingen tid och skriver inga frames.
//! Den logaritmiska töjningen räknas fortfarande från den ursprungliga
//! geometrin. PE redovisas från glödgningen (PE = 0 direkt efter).

use super::api::{StepCtx, StepType};
use crate::fem::model::StepDef;
use crate::model::{fail, FemError, Result};

pub struct Anneal;

pub static STEP_TYPES: &[&dyn StepType] = &[&Anneal];

impl StepType for Anneal {
    fn name(&self) -> &'static str {
        "ANNEAL"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "ANNEAL").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if kw.has("TEMPERATURE") {
            return kw.err("TEMPERATURE= stöds inte ännu");
        }
        if ctx.step.perturbation {
            return kw.err("*ANNEAL kan inte vara ett perturbationssteg");
        }
        let p = ctx.prep;
        let m = p.model;
        for e in 0..m.elements.len() {
            let ectx = p.ctx(e, ctx.nlgeom, ctx.state.time, 0.0);
            let Some(mat) = ectx.material else { continue };
            let et = m.elements[e].etype;
            let st = &ctx.state.elem_state[e];
            let nip = et.n_ips();
            if st.is_empty() || nip == 0 || st.len() % nip != 0 {
                continue;
            }
            let stride = st.len() / nip;
            let Some(ncomp) = stride.checked_sub(mat.n_state()).filter(|n| [1, 3, 4, 6].contains(n)) else {
                return fail(format!("{}: *ANNEAL stöds inte för elementtypen {}", m.elem_name(e), et.name()));
            };
            // aktuell total töjning per integrationspunkt (logaritmisk vid NLGEOM, annars E)
            let ue = p.gather(e, &ctx.state.u);
            let out = et.output(&ectx, &ue, st).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
            let fld = out.iter().find(|f| f.name == if ctx.nlgeom { "LE" } else { "E" }).or_else(|| out.iter().find(|f| f.name == "E" || f.name == "LE"));
            let Some(fld) = fld else {
                return fail(format!("{}: *ANNEAL kräver töjningsutdata (E eller logaritmisk töjning) för {}", m.elem_name(e), et.name()));
            };
            let names = ["11", "22", "33", "12", "13", "23"];
            let mut st = st.clone();
            for k in 0..nip {
                let mut eps = [0.0; 6];
                if let Some(pt) = fld.points.iter().find(|pt| pt.0 == k + 1).or(fld.points.get(k)) {
                    for (l, v) in fld.labels.iter().zip(&pt.2) {
                        if let Some(i) = names.iter().position(|n| l.ends_with(n)) {
                            eps[i] = *v;
                        }
                    }
                }
                let base = k * stride;
                st[base..base + ncomp].iter_mut().for_each(|x| *x = 0.0);
                if !mat.anneal(&eps, &mut st[base + ncomp..base + stride]) {
                    return fail(format!("{}: materialet stöder inte *ANNEAL", m.elem_name(e)));
                }
            }
            ctx.state.elem_state[e] = st;
        }
        for name in ["V", "A"] {
            if let Some(v) = ctx.state.vectors.get_mut(name) {
                v.iter_mut().for_each(|x| x.iter_mut().for_each(|y| *y = 0.0));
            }
        }
        ctx.state.reaction.iter_mut().for_each(|x| x.iter_mut().for_each(|y| *y = 0.0));
        ctx.out.increments = 1;
        Ok(())
    }
}
