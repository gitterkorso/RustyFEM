//! XFEM-sprickor (`elements/xfem.rs`) som interaktion: återställer sprickan
//! när analysen börjar, låter sprickan växa när ett inkrement har godkänts
//! (`commit`) och ger nodutdata PHILSM och PSILSM.

use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::elements::xfem::{self, Crack};
use crate::fem::dof::NodeVec;
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::{fail, Result};

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["ENRICHMENT"], build }];

fn build(model: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let Some(name) = kw.param_upper("NAME") else { return Ok(Vec::new()) };
    let Some(crack) = xfem::crack_of(model, &name) else {
        return fail(format!("{}: berikningen {name} har inga XFEM-element", kw.location()));
    };
    crack.reset();
    Ok(vec![Box::new(XfemCrack { crack })])
}

struct XfemCrack {
    crack: &'static Crack,
}

impl Interaction for XfemCrack {
    fn name(&self) -> String {
        format!("XFEM {}", self.crack.name)
    }

    fn is_linear(&self) -> bool {
        self.crack.stationary && !self.crack.contact
    }

    fn contribute(&self, ctx: &InteractionCtx, _u: &[NodeVec], _fint: &mut [NodeVec], _k: Option<&mut dyn TangentSink>) -> Result<()> {
        self.crack.set_time(ctx.time);
        Ok(())
    }

    fn commit(&self) {
        if !self.crack.stationary {
            self.crack.propagate();
        }
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let mut phi = NodeOutput { name: "PHILSM".into(), labels: vec!["PHILSM".into()], values: Vec::new() };
        let mut psi = NodeOutput { name: "PSILSM".into(), labels: vec!["PSILSM".into()], values: Vec::new() };
        for (&n, &(p, s)) in &self.crack.nodal {
            phi.values.push((n, vec![p]));
            psi.values.push((n, vec![s]));
        }
        vec![phi, psi]
    }
}
