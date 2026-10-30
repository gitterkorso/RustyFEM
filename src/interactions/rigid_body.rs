//! Stela kroppar: *RIGID BODY, REF NODE=, ELSET=, PIN NSET=, TIE NSET=.
//!
//! Alla noder i kroppen följer referensnoden R som en stel kropp:
//! u = u_R + (R(θ_R) − I)(X − X_R) för translationerna och θ = θ_R för
//! rotationerna. Noderna är
//! - noderna i ELSET= (stela element R3D4/R3D3/R2D2/RAX2 eller deformerbara
//!   element som görs stela): translationer och, om noden har dem, rotationer,
//! - PIN NSET=: bara translationer (rotationerna är fria),
//! - TIE NSET=: translationer och rotationer.
//!
//! Referensnoden får translationer och rotationer (UR3 i 2D, UR1–UR3 i 3D).
//! Axisymmetriska kroppar (RAX2) behandlas som plana stela kroppar med UR3,
//! (kontrollerat: utan randvillkor på UR3 kan kroppen
//! vrida sig i rz-planet).
//!
//! Villkoren elimineras exakt av kärnan (`fem/mpc.rs`, master–slav):
//! - utan NLGEOM linjäriserat, u = u_R + θ_R × r₀ (r₀ = X − X_R), dvs.
//!   NLGEOM=NO (kontrollerat i tests/cases/rigid_user),
//! - vid NLGEOM linjäriseras villkoret om i varje iteration
//!   (`linear_constraints_at`): δu = δu_R + δw × a med a = R(θ_R) r₀ och
//!   δw = rumsligt spinn (samma uppdatering R ← exp([δw]) R som NLGEOM-steget
//!   använder), och rhs väljs så att slavens läge blir exakt
//!   u = u_R + a − r₀. Tangenten saknar termen ∂T/∂u · f (villkorskrafternas
//!   geometriska styvhet), så konvergensen är linjär när kroppen roterar under
//!   stor last; jämviktsläget påverkas inte.
//!
//! Övrigt:
//! - ANALYTICAL SURFACE=: en analytisk yta (`interactions/analytical.rs`) som
//!   följer referensnoden; utan andra noder ger kroppen inga villkor.
//! - Inne i en *PART syftar REF NODE, ELSET, PIN/TIE NSET och ANALYTICAL
//!   SURFACE på partens noder och set; kroppen skapas en gång per instans
//!   (`_INSTANCE` från *INSTANCE, se `inp/keywords/assembly.rs`).
//! - POSITION=CENTER OF MASS: referensnoden flyttas till masscentrum före
//!   analysen (`reposition`, anropas när filen är läst) av elementens
//!   fördelade massa (ρ · ∫Nᵢ dV för deformerbara element med densitet, ρ · t ·
//!   ∫Nᵢ dA för stela element med DENSITY= och tjocklek/area på dataraden).
//!   Punktmassor (*MASS) ingår inte.
//! - DENSITY= (stela element) ger nodmassor (`Interaction::mass`); den
//!   gäller bara explicit analys och påverkar inte implicit
//!   *DYNAMIC (men masscentrum för POSITION=CENTER OF MASS).
//! - Ändliga rotationer även i *DYNAMIC med NLGEOM (`steps/dynamic.rs`
//!   linjäriserar villkoren om i varje iteration).

use super::analytical::scoped;
use super::geom;
use super::{Interaction, InteractionCtx, InteractionFactory, LinearConstraint, TangentSink};
use crate::elements::common::sub;
use crate::fem::dof::{Dof, NodeVec, UR1, UR3};
use crate::fem::model::{FeModel, RawKeyword};
use crate::kinematics::tensor;
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["RIGID BODY"], build }];

/// En nod i den stela kroppen.
#[derive(Debug, Clone)]
struct Member {
    node: usize,
    /// Läge relativt referensnoden i referensläget.
    r0: [f64; 3],
    /// Translationer som följer kroppen (i modellens dimension och aktiva för noden).
    trans: Vec<Dof>,
    /// Rotationer som följer kroppen.
    rot: Vec<Dof>,
}

pub struct RigidBody {
    name: String,
    refn: usize,
    members: Vec<Member>,
    dim: usize,
    /// Massa från stela element med DENSITY= (nod, massa), för *DYNAMIC, EXPLICIT.
    masses: Vec<(usize, f64)>,
}

fn single_node(m: &FeModel, kw: &RawKeyword, key: &str) -> Result<usize> {
    let r = crate::inp::required(kw, key)?;
    // inne i en part syftar namnen på instansens noder och set
    let v = geom::node_ref(m, kw, &scoped(kw, r))?;
    if v.len() != 1 {
        return kw.err(format!("{key}={r} ska vara en enda nod"));
    }
    Ok(v[0])
}

/// Elementen i ELSET= (med instansprefix inne i en part).
fn body_elements(m: &FeModel, kw: &RawKeyword) -> Result<Vec<usize>> {
    match kw.param("ELSET") {
        Some(es) => geom::elem_ref(m, kw, &scoped(kw, es)),
        None => Ok(Vec::new()),
    }
}

fn build(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    if kw.has("_PART") && !kw.has("_INSTANCE") {
        // parten används inte i någon instans
        return Ok(Vec::new());
    }
    let refn = single_node(m, kw, "REF NODE")?;
    let dim = m.dim();
    let active = geom::element_active(m);
    let trans = geom::trans_dofs(dim);
    let rots = geom::rot_dofs(dim);
    // (nod, med rotationer)
    let mut nodes: Vec<(usize, bool)> = Vec::new();
    let elems = body_elements(m, kw)?;
    if let Some(inst) = kw.param("_INSTANCE").and_then(|i| m.instance_index(&i.to_uppercase())) {
        // en stel kropp i en part ska omfatta partens alla element
        if m.elements.iter().enumerate().any(|(e, el)| el.inst == inst && elems.binary_search(&e).is_err() && !elems.contains(&e)) {
            return kw.err("en *RIGID BODY i en part ska omfatta alla partens element (ELSET=); definiera kroppen på modellnivå om bara en del av parten är stel");
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
            // bara en analytisk yta: kontakten (interactions/analytical.rs) aktiverar referensnoden
            return Ok(Vec::new());
        }
        return kw.err("kroppen saknar noder (ange ELSET=, PIN NSET=, TIE NSET= eller ANALYTICAL SURFACE=)");
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

/// Nodmassor från stela element i kroppen med DENSITY= och tjocklek
/// (R3D4/R3D3/RAX2) eller tvärsnittsarea (R2D2) på dataraden: ρ · t · ∫Nᵢ dA
/// (RAX2 med 2π r).
fn rigid_element_masses(m: &FeModel, kw: &RawKeyword, elems: &[usize]) -> Result<Vec<(usize, f64)>> {
    let Some(rho) = kw.param("DENSITY") else { return Ok(Vec::new()) };
    let Some(rho) = crate::fem::model::parse_f64(rho) else { return kw.err("DENSITY= ska vara ett tal") };
    if kw.has("NODAL THICKNESS") {
        return kw.err("NODAL THICKNESS stöds inte ännu");
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

/// Nodmassor (translation) för masscentrum: deformerbara element med densitet
/// (ρ · ∫Nᵢ dV) och stela element med DENSITY=.
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

/// POSITION=CENTER OF MASS: flyttar referensnoderna till kropparnas
/// masscentrum (anropas när .inp-filen är läst, före analysen).
pub fn reposition(m: &mut FeModel) -> Result<()> {
    let kws: Vec<RawKeyword> = m.extra.iter().filter(|k| k.name == "RIGID BODY" && k.param_upper("POSITION").is_some_and(|p| p != "INPUT")).cloned().collect();
    for kw in kws {
        if kw.param_upper("POSITION").as_deref() != Some("CENTER OF MASS") {
            return kw.err("POSITION ska vara INPUT eller CENTER OF MASS");
        }
        if kw.has("_PART") && !kw.has("_INSTANCE") {
            continue;
        }
        let refn = single_node(m, &kw, "REF NODE")?;
        let ms = body_masses(m, &kw)?;
        let tot: f64 = ms.iter().map(|p| p.1).sum();
        if tot <= 0.0 {
            // ingen massa: referensnoden flyttas inte
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

    /// Villkoren linjäriserade kring referensnodens rotation (rotationsvektor
    /// `theta`, total). `exact`: rhs så att slavens läge blir exakt (NLGEOM),
    /// annars de rena linjära villkoren u = u_R + θ_R × r₀ (rhs = 0).
    fn constraints(&self, theta: [f64; 3], u: Option<&[NodeVec]>) -> Vec<LinearConstraint> {
        let r = self.refn;
        let rot = tensor::exp_so3(theta);
        let mut out = Vec::new();
        for mb in &self.members {
            // a = R r₀ (aktuellt läge relativt referensnoden)
            let a = if u.is_some() { tensor::mv(&rot, mb.r0) } else { mb.r0 };
            for &d in &mb.trans {
                let i = (d - 1) as usize;
                // u_i − u_R,i − (δw × a)_i-termer = rhs
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
                    // exakt läge: u_i = u_R,i + a_i − r₀,i
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

    /// Stela elements massa (DENSITY=) i deras noder (translationer).
    fn mass(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        let mut v = Vec::new();
        for &(n, mm) in &self.masses {
            for &d in geom::trans_dofs(self.dim) {
                v.push((n, d, n, d, mm));
            }
        }
        v
    }

    /// Villkorskrafternas geometriska styvhet vid NLGEOM (termen ∂Tᵀ/∂q · f som
    /// master–slav-elimineringen saknar): med slavkraften f (elementens inre
    /// kraft i noden, som redan är assemblerad när interaktionerna anropas) och
    /// a = R r₀ ger andra variationen f · (δw₁ × (δw₂ × a)) den symmetriserade
    /// matrisen ½(a fᵀ + f aᵀ) − (f · a) I på referensnodens rotationer.
    /// Påverkar bara konvergensen, inte jämviktsläget.
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
