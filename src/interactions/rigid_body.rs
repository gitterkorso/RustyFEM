//! Rigid bodies: *RIGID BODY, REF NODE=, ELSET=, PIN NSET=, TIE NSET=.
//!
//! All nodes of the body follow the reference node R as a rigid body:
//! u = u_R + (R(θ_R) − I)(X − X_R) for the translations and θ = θ_R for the
//! rotations. The nodes are
//! - the nodes of ELSET= (rigid elements R3D4/R3D3/R2D2/RAX2 or deformable
//!   elements made rigid): translations and, if the node has them, rotations,
//! - PIN NSET=: translations only (the rotations are free),
//! - TIE NSET=: translations and rotations.
//!
//! The reference node gets translations and rotations (UR3 in 2D, UR1–UR3 in 3D).
//! Axisymmetric bodies (RAX2) are treated as planar rigid bodies with UR3
//! (verified: without a boundary condition on UR3 the body can
//! rotate in the rz plane).
//!
//! The constraints are eliminated exactly by the core (`fem/mpc.rs`, master–slave):
//! - without NLGEOM linearized, u = u_R + θ_R × r₀ (r₀ = X − X_R), i.e.
//!   NLGEOM=NO (verified in tests/cases/rigid_user),
//! - with NLGEOM the constraint is relinearized in every iteration
//!   (`linear_constraints_at`): δu = δu_R + δw × a with a = R(θ_R) r₀ and
//!   δw = spatial spin (the same update R ← exp([δw]) R that the NLGEOM step
//!   uses), and rhs is chosen so that the slave position is exact:
//!   u = u_R + a − r₀. The tangent lacks the term ∂T/∂u · f (geometric
//!   stiffness of the constraint forces), so convergence is linear when the
//!   body rotates under large load; the equilibrium state is not affected.
//!
//! Other:
//! - ANALYTICAL SURFACE=: an analytical surface (`interactions/analytical.rs`)
//!   that follows the reference node; without other nodes the body gives no
//!   constraints.
//! - Inside a *PART, REF NODE, ELSET, PIN/TIE NSET and ANALYTICAL SURFACE
//!   refer to the part's nodes and sets; the body is created once per instance
//!   (`_INSTANCE` from *INSTANCE, see `inp/keywords/assembly.rs`).
//! - POSITION=CENTER OF MASS: the reference node is moved to the center of mass
//!   before the analysis (`reposition`, called once the file has been read) of
//!   the elements' distributed mass (ρ · ∫Nᵢ dV for deformable elements with a
//!   density, ρ · t · ∫Nᵢ dA for rigid elements with DENSITY= and
//!   thickness/area on the data line). Point masses (*MASS) are not included.
//! - DENSITY= (rigid elements) gives nodal masses (`Interaction::mass`); it
//!   only applies to explicit analysis and does not affect implicit
//!   *DYNAMIC (but does affect the center of mass for POSITION=CENTER OF MASS).
//! - Finite rotations also in *DYNAMIC with NLGEOM (`steps/dynamic.rs`
//!   relinearizes the constraints in every iteration).

use super::analytical::scoped;
use super::geom;
use super::{Interaction, InteractionCtx, InteractionFactory, LinearConstraint, TangentSink};
use crate::elements::common::sub;
use crate::fem::dof::{Dof, NodeVec, UR1, UR3};
use crate::fem::model::{FeModel, RawKeyword};
use crate::kinematics::tensor;
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["RIGID BODY"], build }];

/// A node of the rigid body.
#[derive(Debug, Clone)]
struct Member {
    node: usize,
    /// Position relative to the reference node in the reference configuration.
    r0: [f64; 3],
    /// Translations that follow the body (in the model dimension and active for the node).
    trans: Vec<Dof>,
    /// Rotations that follow the body.
    rot: Vec<Dof>,
}

pub struct RigidBody {
    name: String,
    refn: usize,
    members: Vec<Member>,
    dim: usize,
    /// Mass from rigid elements with DENSITY= (node, mass), for *DYNAMIC, EXPLICIT.
    masses: Vec<(usize, f64)>,
}

fn single_node(m: &FeModel, kw: &RawKeyword, key: &str) -> Result<usize> {
    let r = crate::inp::required(kw, key)?;
    // inside a part the names refer to the instance's nodes and sets
    let v = geom::node_ref(m, kw, &scoped(kw, r))?;
    if v.len() != 1 {
        return kw.err(format!("{key}={r} must be a single node"));
    }
    Ok(v[0])
}

/// The elements of ELSET= (with instance prefix inside a part).
fn body_elements(m: &FeModel, kw: &RawKeyword) -> Result<Vec<usize>> {
    match kw.param("ELSET") {
        Some(es) => geom::elem_ref(m, kw, &scoped(kw, es)),
        None => Ok(Vec::new()),
    }
}

fn build(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    if kw.has("_PART") && !kw.has("_INSTANCE") {
        // the part is not used in any instance
        return Ok(Vec::new());
    }
    let refn = single_node(m, kw, "REF NODE")?;
    let dim = m.dim();
    let active = geom::element_active(m);
    let trans = geom::trans_dofs(dim);
    let rots = geom::rot_dofs(dim);
    // (node, with rotations)
    let mut nodes: Vec<(usize, bool)> = Vec::new();
    let elems = body_elements(m, kw)?;
    if let Some(inst) = kw.param("_INSTANCE").and_then(|i| m.instance_index(&i.to_uppercase())) {
        // a rigid body in a part must include all elements of the part
        if m.elements.iter().enumerate().any(|(e, el)| el.inst == inst && elems.binary_search(&e).is_err() && !elems.contains(&e)) {
            return kw.err("a *RIGID BODY in a part must include all elements of the part (ELSET=); define the body at model level if only some of the part is rigid");
        }
    }
    for &e in &elems {
        for &n in &m.elements[e].nodes {
            nodes.push((n, true));
        }
    }
    if let Some(ns) = kw.param("TIE NSET") {
        nodes.extend(geom::node_ref(m, kw, &scoped(kw, ns))?.into_iter().map(|n| (n, true)));
    }
    if let Some(ns) = kw.param("PIN NSET") {
        nodes.extend(geom::node_ref(m, kw, &scoped(kw, ns))?.into_iter().map(|n| (n, false)));
    }
    if nodes.is_empty() {
        if kw.has("ANALYTICAL SURFACE") {
            // only an analytical surface: the contact (interactions/analytical.rs) activates the reference node
            return Ok(Vec::new());
        }
        return kw.err("the body has no nodes (give ELSET=, PIN NSET=, TIE NSET= or ANALYTICAL SURFACE=)");
    }
    let masses = rigid_element_masses(m, kw, &elems)?;
    nodes.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    nodes.dedup_by_key(|p| p.0);
    let mut members = Vec::new();
    for (n, with_rot) in nodes {
        if n == refn {
            continue;
        }
        let t: Vec<Dof> = trans.iter().copied().filter(|&d| geom::has(&active, n, d)).collect();
        let r: Vec<Dof> = if with_rot { rots.iter().copied().filter(|&d| geom::has(&active, n, d)).collect() } else { Vec::new() };
        members.push(Member { node: n, r0: sub(m.nodes[n], m.nodes[refn]), trans: t, rot: r });
    }
    Ok(vec![Box::new(RigidBody { name: format!("*RIGID BODY ({})", kw.location()), refn, members, dim, masses })])
}

/// Nodal masses from rigid elements of the body with DENSITY= and thickness
/// (R3D4/R3D3/RAX2) or cross-sectional area (R2D2) on the data line: ρ · t · ∫Nᵢ dA
/// (RAX2 with 2π r).
fn rigid_element_masses(m: &FeModel, kw: &RawKeyword, elems: &[usize]) -> Result<Vec<(usize, f64)>> {
    let Some(rho) = kw.param("DENSITY") else { return Ok(Vec::new()) };
    let Some(rho) = crate::fem::model::parse_f64(rho) else { return kw.err("DENSITY= must be a number") };
    if kw.has("NODAL THICKNESS") {
        return kw.err("NODAL THICKNESS is not supported yet");
    }
    let t = kw.first_floats()?.first().copied().unwrap_or(0.0);
    let mut out: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for &e in elems {
        let el = &m.elements[e];
        if !crate::elements::rigid::is_rigid(el.etype) {
            continue;
        }
        let axi = el.etype.name() == "RAX2";
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
        let f = geom::Facet { elem: e, nodes: el.nodes.clone(), inside: [0.0; 3], dim: el.etype.dim(), thick: 1.0 };
        for (xi, w) in geom::gauss(f.dim, x.len(), 3) {
            let fp = geom::eval(&f, &x, xi);
            let wr = if axi { 2.0 * std::f64::consts::PI * fp.x[0] } else { 1.0 };
            for (k, &n) in el.nodes.iter().enumerate() {
                *out.entry(n).or_default() += rho * t * fp.n[k] * w * fp.jac * wr;
            }
        }
    }
    Ok(out.into_iter().collect())
}

/// Nodal masses (translation) for the center of mass: deformable elements with
/// a density (ρ · ∫Nᵢ dV) and rigid elements with DENSITY=.
fn body_masses(m: &FeModel, kw: &RawKeyword) -> Result<Vec<(usize, f64)>> {
    let elems = body_elements(m, kw)?;
    let mut out: std::collections::BTreeMap<usize, f64> = rigid_element_masses(m, kw, &elems)?.into_iter().collect();
    for &e in &elems {
        let el = &m.elements[e];
        if crate::elements::rigid::is_rigid(el.etype) {
            continue;
        }
        let section = el.section.map(|s| &m.sections[s]);
        let Some(mi) = section.and_then(|s| s.material_index) else { continue };
        let rho = m.materials[mi].behavior("DENSITY").map(|b| b.first_floats()).transpose()?.and_then(|v| v.first().copied()).unwrap_or(0.0);
        if rho == 0.0 {
            continue;
        }
        let mat = crate::materials::build(&m.materials[mi])?;
        let ctx = crate::elements::ElemCtx {
            index: e,
            label: el.label,
            x: el.nodes.iter().map(|&n| m.nodes[n]).collect(),
            section,
            material: Some(&*mat),
            density: rho,
            nlgeom: false,
            time: 0.0,
            dt: 0.0,
            temp: Vec::new(),
            temp_old: Vec::new(),
            orientation: None,
        };
        let w = el.etype.volume_weights(&ctx)?;
        for (k, &n) in el.nodes.iter().enumerate() {
            *out.entry(n).or_default() += rho * w.get(k).copied().unwrap_or(0.0);
        }
    }
    Ok(out.into_iter().collect())
}

/// POSITION=CENTER OF MASS: moves the reference nodes to the bodies' centers
/// of mass (called once the .inp file has been read, before the analysis).
pub fn reposition(m: &mut FeModel) -> Result<()> {
    let kws: Vec<RawKeyword> = m.extra.iter().filter(|k| k.name == "RIGID BODY" && k.param_upper("POSITION").is_some_and(|p| p != "INPUT")).cloned().collect();
    for kw in kws {
        if kw.param_upper("POSITION").as_deref() != Some("CENTER OF MASS") {
            return kw.err("POSITION must be INPUT or CENTER OF MASS");
        }
        if kw.has("_PART") && !kw.has("_INSTANCE") {
            continue;
        }
        let refn = single_node(m, &kw, "REF NODE")?;
        let ms = body_masses(m, &kw)?;
        let tot: f64 = ms.iter().map(|p| p.1).sum();
        if tot <= 0.0 {
            // no mass: the reference node is not moved
            continue;
        }
        let mut c = [0.0; 3];
        for &(n, w) in &ms {
            for k in 0..3 {
                c[k] += w * m.nodes[n][k] / tot;
            }
        }
        m.nodes[refn] = c;
    }
    Ok(())
}

impl RigidBody {
    fn ref_dofs(&self) -> Vec<Dof> {
        let mut v = geom::trans_dofs(self.dim).to_vec();
        v.extend(geom::rot_dofs(self.dim));
        v
    }

    /// The constraints linearized about the reference node rotation (rotation
    /// vector `theta`, total). `exact`: rhs so that the slave position is exact
    /// (NLGEOM), otherwise the plain linear constraints u = u_R + θ_R × r₀ (rhs = 0).
    fn constraints(&self, theta: [f64; 3], u: Option<&[NodeVec]>) -> Vec<LinearConstraint> {
        let r = self.refn;
        let rot = tensor::exp_so3(theta);
        let mut out = Vec::new();
        for mb in &self.members {
            // a = R r₀ (current position relative to the reference node)
            let a = if u.is_some() { tensor::mv(&rot, mb.r0) } else { mb.r0 };
            for &d in &mb.trans {
                let i = (d - 1) as usize;
                // u_i − u_R,i − (δw × a)_i terms = rhs
                let mut t = vec![(mb.node, d, 1.0), (r, d, -1.0)];
                if self.dim == 3 {
                    // (w × a)_i = w_j a_k − w_k a_j
                    let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                    t.push((r, UR1 + j as Dof, -a[k]));
                    t.push((r, UR1 + k as Dof, a[j]));
                } else if i == 0 {
                    // w e3 × a = (−w a2, w a1)
                    t.push((r, UR3, a[1]));
                } else {
                    t.push((r, UR3, -a[0]));
                }
                let rhs = match u {
                    // exact position: u_i = u_R,i + a_i − r₀,i
                    Some(u) => {
                        let mut v = a[i] - mb.r0[i];
                        for &(n, dd, c) in &t[2..] {
                            v += c * u[n][(dd - 1) as usize];
                        }
                        v
                    }
                    None => 0.0,
                };
                out.push(LinearConstraint { terms: t.into_iter().filter(|p| p.2 != 0.0).collect(), rhs, force_slave: false });
            }
            for &d in &mb.rot {
                out.push(LinearConstraint { terms: vec![(mb.node, d, 1.0), (r, d, -1.0)], rhs: 0.0, force_slave: false });
            }
        }
        out
    }
}

impl Interaction for RigidBody {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        self.ref_dofs().into_iter().map(|d| (self.refn, d)).collect()
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.members.iter().map(|mb| vec![self.refn, mb.node]).collect()
    }

    fn linear_constraints(&self) -> Result<Vec<LinearConstraint>> {
        Ok(self.constraints([0.0; 3], None))
    }

    fn constraints_depend_on_u(&self) -> bool {
        true
    }

    fn linear_constraints_at(&self, u: &[NodeVec], nlgeom: bool) -> Result<Vec<LinearConstraint>> {
        if !nlgeom {
            return self.linear_constraints();
        }
        let ur = &u[self.refn];
        let theta = if self.dim == 2 { [0.0, 0.0, ur[5]] } else { [ur[3], ur[4], ur[5]] };
        Ok(self.constraints(theta, Some(u)))
    }

    fn is_linear(&self) -> bool {
        true
    }

    /// Mass of rigid elements (DENSITY=) at their nodes (translations).
    fn mass(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        let mut v = Vec::new();
        for &(n, mm) in &self.masses {
            for &d in geom::trans_dofs(self.dim) {
                v.push((n, d, n, d, mm));
            }
        }
        v
    }

    /// Geometric stiffness of the constraint forces with NLGEOM (the term
    /// ∂Tᵀ/∂q · f that the master–slave elimination lacks): with the slave force
    /// f (the elements' internal force at the node, already assembled when the
    /// interactions are called) and a = R r₀, the second variation
    /// f · (δw₁ × (δw₂ × a)) gives the symmetrized matrix ½(a fᵀ + f aᵀ) − (f · a) I
    /// on the reference node rotations. Only affects convergence, not the
    /// equilibrium state.
    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], k: Option<&mut dyn TangentSink>) -> Result<()> {
        let Some(k) = k else { return Ok(()) };
        if !ctx.nlgeom {
            return Ok(());
        }
        let ur = &u[self.refn];
        let theta = if self.dim == 2 { [0.0, 0.0, ur[5]] } else { [ur[3], ur[4], ur[5]] };
        let rot = tensor::exp_so3(theta);
        let mut kg = [[0.0; 3]; 3];
        for mb in &self.members {
            let a = tensor::mv(&rot, mb.r0);
            let mut f = [0.0; 3];
            for &d in &mb.trans {
                f[(d - 1) as usize] = fint[mb.node][(d - 1) as usize];
            }
            let fa = f[0] * a[0] + f[1] * a[1] + f[2] * a[2];
            for i in 0..3 {
                for j in 0..3 {
                    kg[i][j] += 0.5 * (a[i] * f[j] + f[i] * a[j]) - if i == j { fa } else { 0.0 };
                }
            }
        }
        let rd: &[Dof] = if self.dim == 2 { &[UR3] } else { &[UR1, UR1 + 1, UR3] };
        for &di in rd {
            for &dj in rd {
                let v = kg[(di - UR1) as usize][(dj - UR1) as usize];
                if v != 0.0 {
                    k.add(self.refn, di, self.refn, dj, v);
                }
            }
        }
        Ok(())
    }
}
