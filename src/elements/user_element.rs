//! Användarelement: *USER ELEMENT, TYPE=Un, NODES=, COORDINATES=,
//! PROPERTIES=, I PROPERTIES=, VARIABLES=, UNSYMM och *UEL PROPERTY, ELSET=.
//!
//! Elementtypen skapas när *ELEMENT, TYPE=Un läses (`lookup`, anropas från
//! `inp/keywords/element.rs`) från den *USER ELEMENT som definierar typen,
//! och anropar en UEL (`crate::user::uel`) för typnumret n. Dataraderna i
//! *USER ELEMENT anger de aktiva frihetsgraderna: första raden gäller från
//! nod 1, en senare rad "k, d1, d2, ..." från nod k och framåt.
//!
//! *UEL PROPERTY sparas som ett tvärsnitt (`SectionKind::Other`) så att
//! `FeModel::finalize` kopplar det till elementen; dataraderna är först
//! PROPERTIES reella värden och sedan I PROPERTIES heltal.
//!
//! Elementets tillstånd är SVARS(1..NSVARS) följt av ENERGY(1..8). AMATRX
//! symmetriseras (½(K + Kᵀ)) eftersom systemmatrisen är symmetrisk.
//!
//! Begränsningar: inga integrationspunktsutdata (inga S
//! för användarelement), inga fördelade laster (*DLOAD), FILE= och LINEAR
//! (linjära användarelement från fil) stöds inte.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, MassKind};
use crate::fem::dof::Dof;
use crate::fem::model::RawKeyword;
use crate::inp::{parse_label, InpCtx};
use crate::model::{fail, FemError, Result};
use crate::user::uel::{self, UelArgs};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Användarelement skapas från .inp-filen; inga statiska typer.
pub static TYPES: &[&dyn ElementType] = &[];

pub struct UserElement {
    name: &'static str,
    jtype: u32,
    nnode: usize,
    mcrd: usize,
    nprops: usize,
    njprops: usize,
    nsvars: usize,
    dofs: Vec<&'static [Dof]>,
    dim: usize,
}

/// Elementtypen för TYPE=`ty` (t.ex. "U1") från *USER ELEMENT i parten eller
/// modellen som läses, eller `None` om typen inte är ett användarelement.
pub fn lookup(ctx: &InpCtx, ty: &str) -> Result<Option<&'static dyn ElementType>> {
    let up = ty.trim().to_uppercase();
    let Some(n) = up.strip_prefix('U').and_then(|s| s.parse::<u32>().ok()) else { return Ok(None) };
    let found = ctx
        .mesh_ref()
        .extra
        .iter()
        .chain(ctx.model.extra.iter())
        .find(|k| k.name == "USER ELEMENT" && k.param_upper("TYPE").as_deref() == Some(up.as_str()));
    let Some(kw) = found else { return Ok(None) };
    Ok(Some(define(kw, &up, n)?))
}

/// Tolkar *USER ELEMENT och skapar (eller återanvänder) elementtypen.
pub fn define(kw: &RawKeyword, name: &str, jtype: u32) -> Result<&'static UserElement> {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static UserElement>>> = OnceLock::new();
    let key = format!("{:?}|{:?}", kw.params, kw.data.iter().map(|d| &d.fields).collect::<Vec<_>>());
    let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = cache.get(&key) {
        return Ok(*t);
    }
    for p in ["FILE", "LINEAR"] {
        if kw.has(p) {
            return kw.err(format!("{p} (linjära användarelement från fil) stöds inte ännu"));
        }
    }
    let int = |key: &str, default: usize| -> Result<usize> {
        match kw.param(key) {
            None => Ok(default),
            Some(v) => parse_label(v).filter(|&x| x >= 0).map(|x| x as usize).ok_or_else(|| FemError(format!("{}: {key}={v} ska vara ett heltal", kw.location()))),
        }
    };
    let nnode = int("NODES", 0)?;
    if nnode == 0 {
        return kw.err("NODES= saknas");
    }
    let mcrd = int("COORDINATES", 1)?.clamp(1, 3);
    let (nprops, njprops, nsvars) = (int("PROPERTIES", 0)?, int("I PROPERTIES", 0)?, int("VARIABLES", 1)?);
    // aktiva frihetsgrader per nod
    let mut per_node: Vec<Option<Vec<Dof>>> = vec![None; nnode];
    for (i, d) in kw.records().iter().enumerate() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.is_empty() {
            continue;
        }
        let nums: Vec<i64> = f.iter().map(|s| parse_label(s).ok_or_else(|| FemError(format!("{} rad {}: '{s}' är inget heltal", kw.file, d.line)))).collect::<Result<_>>()?;
        let (start, list) = if i == 0 { (1, &nums[..]) } else { (nums[0], &nums[1..]) };
        if start < 1 || start as usize > nnode {
            return kw.err_at(d.line, format!("nodposition {start} finns inte (NODES={nnode})"));
        }
        let mut dofs: Vec<Dof> = Vec::new();
        for &x in list {
            match x {
                1..=7 | 11 => dofs.push(x as Dof),
                _ => return kw.err_at(d.line, format!("frihetsgrad {x} stöds inte (1–7, 11)")),
            }
        }
        dofs.sort_unstable();
        dofs.dedup();
        for slot in per_node.iter_mut().skip(start as usize - 1) {
            *slot = Some(dofs.clone());
        }
    }
    let mut dofs: Vec<&'static [Dof]> = Vec::with_capacity(nnode);
    for (k, d) in per_node.into_iter().enumerate() {
        let Some(d) = d else { return kw.err(format!("aktiva frihetsgrader saknas för nod {}", k + 1)) };
        dofs.push(Box::leak(d.into_boxed_slice()));
    }
    let dim = if mcrd >= 3 || dofs.iter().any(|d| d.iter().any(|&x| x == 3 || x == 4 || x == 5)) { 3 } else { 2 };
    let t: &'static UserElement = Box::leak(Box::new(UserElement { name: Box::leak(name.to_string().into_boxed_str()), jtype, nnode, mcrd, nprops, njprops, nsvars, dofs, dim }));
    cache.insert(key, t);
    Ok(t)
}

/// Vad UEL:en ska beräkna (LFLAGS(3)).
#[derive(Clone, Copy, PartialEq)]
enum What {
    /// RHS och styvhet.
    Full = 1,
    Stiffness = 2,
    Mass = 4,
    Rhs = 5,
}

/// Resultat av ett UEL-anrop: (inre kraft = −RHS, AMATRX radvis symmetriserad, SVARS + ENERGY).
type Call = (Vec<f64>, Vec<f64>, Vec<f64>);

impl UserElement {
    /// Egenskaper (PROPS, JPROPS) från elementets *UEL PROPERTY.
    fn props(&self, ctx: &ElemCtx) -> Result<(Vec<f64>, Vec<i32>)> {
        if self.nprops == 0 && self.njprops == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let Some(sec) = ctx.section.filter(|s| s.raw.name == "UEL PROPERTY") else {
            return fail(format!("element {} ({}) saknar *UEL PROPERTY", ctx.label, self.name));
        };
        let kw = &sec.raw;
        let recs = kw.records();
        let fields: Vec<&str> = recs.iter().flat_map(|d| d.fields.iter().map(|s| s.trim())).filter(|s| !s.is_empty()).collect();
        if fields.len() < self.nprops + self.njprops {
            return kw.err(format!("PROPERTIES={} och I PROPERTIES={} men dataraderna har {} värden", self.nprops, self.njprops, fields.len()));
        }
        let mut p = Vec::with_capacity(self.nprops);
        for s in &fields[..self.nprops] {
            p.push(crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: '{s}' är inget tal", kw.location())))?);
        }
        let mut j = Vec::with_capacity(self.njprops);
        for s in &fields[self.nprops..self.nprops + self.njprops] {
            j.push(parse_label(s).ok_or_else(|| FemError(format!("{}: '{s}' är inget heltal", kw.location())))? as i32);
        }
        Ok((p, j))
    }

    fn call(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], state: &[f64], what: What) -> Result<Call> {
        let Some(routine) = uel::find(self.jtype)? else {
            return fail(format!(
                "ingen UEL för {}: registrera en i Rust med rustyfem::user::uel::register (exempel: U9001 stång, U9002 fyrhörning) eller ange ett bibliotek med uel_ i RUSTYFEM_USER",
                self.name
            ));
        };
        let (props, jprops) = self.props(ctx)?;
        let n = self.n_dofs();
        let mut rhs = vec![0.0; n];
        let mut amatrx = vec![0.0; n * n];
        let mut svars = vec![0.0; self.nsvars];
        let mut energy = [0.0; 8];
        if state.len() == self.nsvars + 8 {
            svars.copy_from_slice(&state[..self.nsvars]);
            energy.copy_from_slice(&state[self.nsvars..]);
        }
        let coords: Vec<f64> = ctx.x.iter().flat_map(|x| x[..self.mcrd].to_vec()).collect();
        let mut predef = vec![0.0; 2 * self.nnode];
        if ctx.temp.len() == self.nnode && ctx.temp_old.len() == self.nnode {
            for k in 0..self.nnode {
                predef[2 * k] = ctx.temp[k];
                predef[2 * k + 1] = ctx.temp[k] - ctx.temp_old[k];
            }
        }
        let zeros = vec![0.0; n];
        let mut pnewdt = 1.0;
        let t0 = ctx.time - ctx.dt;
        let mut a = UelArgs {
            rhs: &mut rhs,
            amatrx: &mut amatrx,
            svars: &mut svars,
            energy: &mut energy,
            ndofel: n,
            nrhs: 1,
            nsvars: self.nsvars,
            props: &props,
            jprops: &jprops,
            coords: &coords,
            mcrd: self.mcrd,
            nnode: self.nnode,
            u,
            du,
            v: &zeros,
            a: &zeros,
            jtype: self.jtype,
            time: [t0, t0],
            dtime: ctx.dt,
            kstep: 0,
            kinc: 0,
            jelem: ctx.label,
            params: [0.0; 3],
            predef: &predef,
            npredf: 1,
            lflags: [if what == What::Mass { 41 } else { 1 }, ctx.nlgeom as i32, what as i32, 0, 0],
            mlvarx: n,
            pnewdt: &mut pnewdt,
            period: 0.0,
        };
        routine.uel(&mut a).map_err(|e| FemError(format!("{} ({}): {e}", routine.name(), self.name)))?;
        if pnewdt < 1.0 {
            return fail(format!("UEL för {} begär mindre inkrement (PNEWDT = {pnewdt})", self.name));
        }
        let fint: Vec<f64> = rhs.iter().map(|r| -r).collect();
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = 0.5 * (amatrx[i + j * n] + amatrx[j + i * n]);
            }
        }
        let mut st = svars;
        st.extend_from_slice(&energy);
        Ok((fint, k, st))
    }
}

impl ElementType for UserElement {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        self.nnode
    }
    fn node_dofs(&self, local_node: usize) -> &'static [Dof] {
        self.dofs[local_node]
    }

    fn init_state(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(vec![0.0; self.nsvars + 8])
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        let z = vec![0.0; n];
        Ok(self.call(ctx, &z, &z, &[], What::Stiffness)?.1)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let (fint, k, st) = self.call(ctx, u, du, so, if tangent { What::Full } else { What::Rhs })?;
        *sn = st;
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        let z = vec![0.0; n];
        Ok(self.call(ctx, &z, &z, &[], What::Mass)?.1)
    }
}
