//! Område explicit_adv: Euler-element (EC3D8R, `*EULERIAN SECTION`) i
//! explicit analys, med operatordelning i ett Lagrangesteg och en advektion
//! tillbaka till det fasta nätet varje inkrement (Benson 1992, "Computational
//! methods in Lagrangian and Eulerian hydrocodes").
//!
//! Elementet har flera material (tvärsnittets materialinstanser) och tomrum.
//! Varje material i elementet har volymandel φₘ, massa Mₘ, spänning σₘ och
//! materialtillstånd (för *EOS den specifika energin).
//!
//! Lagrangesteget: noderna rör sig med halvstegshastigheten (den allmänna
//! lösaren), hastighetsgradienten L = (1/V) Σ vᵢ ⊗ bᵢ (en punkt, som C3D8R)
//! ger samma töjningsinkrement Δε = sym(L)Δt för alla material i elementet;
//! varje material uppdaterar sin spänning (Jaumann-rotation, *EOS med
//! J = ρ₀/ρ), och materialvolymerna ändras med (1 + tr Δε).
//!
//! Advektionen: för varje elementyta ger den svepta volymen s (ytans fyra
//! fasta och flyttade noder) flödet från uppströmselementet. Materialets
//! volymflöde är s·φₘ på ytan: första ordningen φₘ i uppströmselementet,
//! andra ordningen (standard, `ADVECTION=SECOND ORDER`) med en begränsad
//! lutning (minmod mellan uppströms-, givar- och mottagarelementen över
//! motstående ytor, van Leer-korrektion med Courant-talet). Flödet begränsas
//! så att inget material blir negativt. Massa, spänning·volym och
//! tillstånd·massa följer volymflödet. Rörelsemängden advekteras med
//! elementmittprojektion (standard ELEMENT CENTER PROJECTION):
//! elementets medelhastighet gånger massflödet, och nodernas hastigheter
//! ändras med den massviktade ändringen av elementens medelhastigheter.
//!
//! Krafter på det fasta nätet: fᵢ = bᵢ·σ, σ = Σ φₘ σₘ, och volymviskositet
//! σ_v = ρ Lₑ (b₁ c ε̇ᵥ − b₂² Lₑ ε̇ᵥ² [ε̇ᵥ < 0]) I (ALLVD); nodmassan är
//! elementens massa genom åtta. Tyngdkraft (`*DLOAD`, GRAV) verkar på
//! materialets massa. Stabilitetsgränsen är Lₑ/(c + |v|) med det styvaste
//! materialet i elementet.
//!
//! Utdata: EVF_<instans>, EVF_VOID, SVAVG (volymmedelvärde
//! över materialen), PEEQVAVG och DENSITYVAVG.
//!
//! Begränsningar: ingen hourglass-kontroll (stela nät och endimensionella
//! förlopp påverkas inte), inga Euler-ränder (*EULERIAN BOUNDARY), material
//! som rör sig ut genom en fri rand försvinner, nodernas hastigheter
//! advekteras med första ordningen.

use super::{Eval, Sink};
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::elements::IpField;
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, FemError, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Volymandel under vilken ett material räknas som frånvarande.
pub const VF_TINY: f64 = 1e-6;

/// Platser per material i elementets tillstånd.
const VF: usize = 0;
const MASS: usize = 1;
const S: usize = 2;
const MS: usize = 8;

pub struct EulerMat {
    /// Materialinstansens namn (versaler).
    pub name: String,
    pub model: Arc<dyn MaterialModel>,
    pub rho0: f64,
    pub ns: usize,
}

pub struct Elem {
    pub e: usize,
    pub nodes: [usize; 8],
    /// Materialen (index i `Euler::mats`) i tvärsnittets ordning.
    pub mats: Vec<usize>,
    pub vol0: f64,
    pub le0: f64,
    pub b0: Vec<[f64; 3]>,
    /// Grannen över varje yta (Euler-elementindex, grannens yta).
    pub nbr: [Option<(usize, usize)>; 6],
    pub second_order: bool,
    /// Tyngdkraft per massenhet.
    pub gravity: [f64; 3],
}

pub struct Euler {
    pub mats: Vec<EulerMat>,
    pub elems: Vec<Elem>,
    /// Euler-nodernas index.
    pub nodes: Vec<usize>,
    b1: f64,
    b2: f64,
    /// Nya nodhastigheter efter advektionen (nod, hastighet).
    pending: Vec<(usize, [f64; 3])>,
    /// Elementens töjningshastighet tr L i senaste Lagrangesteget.
    rates: Vec<f64>,
    pub initial_energy: f64,
    pub dt_est: Option<(f64, usize)>,
}

const OPPOSITE: [usize; 6] = [1, 0, 4, 5, 2, 3];

fn stride(m: &EulerMat) -> usize {
    MS + m.ns
}

/// Volymen av hexaedern med noderna `x` (.inp-formatets nodordning).
fn hex_volume(x: &[[f64; 3]; 8]) -> f64 {
    // summa av tetraedrar från tyngdpunkten mot ytornas trianglar
    let c: [f64; 3] = std::array::from_fn(|k| x.iter().map(|p| p[k]).sum::<f64>() / 8.0);
    let mut v = 0.0;
    for f in crate::elements::continuum3d::HEX8_FACES {
        let fc: [f64; 3] = std::array::from_fn(|k| f.iter().map(|&i| x[i][k]).sum::<f64>() / 4.0);
        for t in 0..4 {
            let a = x[f[t]];
            let b = x[f[(t + 1) % 4]];
            v += tet(c, fc, b, a);
        }
    }
    v
}

fn tet(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])) / 6.0
}

/// Volymen som ytan (fyra noder, utåtriktad nodordning) sveper när noderna går
/// från `a` till `b` (positiv utåt).
fn swept(a: &[[f64; 3]; 4], b: &[[f64; 3]; 4]) -> f64 {
    // hexaedern (a-ytan, b-ytan) med a som "botten": nodordningen ger
    // positiv volym när b ligger utanför a-ytans utåtriktade normal
    let x = [a[0], a[3], a[2], a[1], b[0], b[3], b[2], b[1]];
    hex_volume(&x)
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

impl Euler {
    pub fn build(p: &Prepared, step: &StepDef, b1: f64, b2: f64) -> Result<Option<Euler>> {
        let m = p.model;
        let mut mats: Vec<EulerMat> = Vec::new();
        let mut elems: Vec<Elem> = Vec::new();
        let mut of_elem: HashMap<usize, usize> = HashMap::new();
        let mut sec_mats: HashMap<usize, (Vec<usize>, bool)> = HashMap::new();
        for (e, el) in m.elements.iter().enumerate() {
            if el.etype.name() != "EC3D8R" {
                continue;
            }
            let Some(si) = el.section else { return fail(format!("{} saknar tvärsnitt (*EULERIAN SECTION)", m.elem_name(e))) };
            let sec = &m.sections[si];
            if sec.raw.name != "EULERIAN SECTION" {
                return sec.raw.err("EC3D8R kräver *EULERIAN SECTION");
            }
            if !sec_mats.contains_key(&si) {
                let second = match sec.raw.param_upper("ADVECTION").as_deref() {
                    None | Some("SECOND ORDER") => true,
                    Some("FIRST ORDER") => false,
                    Some(o) => return sec.raw.err(format!("ADVECTION={o} stöds inte (FIRST ORDER, SECOND ORDER)")),
                };
                let mut ids = Vec::new();
                for d in &sec.raw.data {
                    let Some(mname) = d.fields.first().map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()) else { continue };
                    let inst = d.fields.get(1).map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).unwrap_or_else(|| mname.clone());
                    let id = match mats.iter().position(|x| x.name == inst) {
                        Some(i) => i,
                        None => {
                            let Some(def) = m.materials.iter().find(|x| x.name == mname) else {
                                return sec.raw.err_at(d.line, format!("materialet '{mname}' är inte definierat"));
                            };
                            let model = crate::materials::build(def)?;
                            let rho0 = def.behavior("DENSITY").map(|k| k.first_floats()).transpose()?.and_then(|v| v.first().copied()).unwrap_or(0.0);
                            if !(rho0 > 0.0) {
                                return sec.raw.err_at(d.line, format!("Euler-materialet '{mname}' saknar *DENSITY"));
                            }
                            let ns = model.n_state();
                            mats.push(EulerMat { name: inst, model, rho0, ns });
                            mats.len() - 1
                        }
                    };
                    ids.push(id);
                }
                sec_mats.insert(si, (ids, second));
            }
            let (ids, second) = sec_mats[&si].clone();
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            let Some(g) = super::super::dt::geometry(crate::elements::Family::Continuum3D, "C3D8R", &x) else {
                return fail(format!("{}: elementet är inverterat eller degenererat", m.elem_name(e)));
            };
            of_elem.insert(e, elems.len());
            elems.push(Elem {
                e,
                nodes: std::array::from_fn(|k| el.nodes[k]),
                mats: ids,
                vol0: g.vol,
                le0: g.lchar,
                b0: g.b,
                nbr: [None; 6],
                second_order: second,
                gravity: [0.0; 3],
            });
        }
        if elems.is_empty() {
            return Ok(None);
        }
        // grannar över ytorna
        let mut faces: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
        for k in 0..elems.len() {
            for (f, fnodes) in crate::elements::continuum3d::HEX8_FACES.iter().enumerate() {
                let mut key: Vec<usize> = fnodes.iter().map(|&i| elems[k].nodes[i]).collect();
                key.sort_unstable();
                if let Some(&(k2, f2)) = faces.get(&key) {
                    elems[k].nbr[f] = Some((k2, f2));
                    elems[k2].nbr[f2] = Some((k, f));
                } else {
                    faces.insert(key, (k, f));
                }
            }
        }
        // tyngdkraft (*DLOAD, GRAV) på Euler-elementen
        for blk in &step.dloads {
            for dl in &blk.items {
                if let Some(&k) = of_elem.get(&dl.elem) {
                    if dl.kind == "GRAV" {
                        let d = [dl.extra.first().copied().unwrap_or(0.0), dl.extra.get(1).copied().unwrap_or(0.0), dl.extra.get(2).copied().unwrap_or(0.0)];
                        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if l > 0.0 {
                            for c in 0..3 {
                                elems[k].gravity[c] += dl.magnitude * d[c] / l;
                            }
                        }
                    } else if matches!(dl.kind.as_str(), "BX" | "BY" | "BZ") {
                        return fail(format!("{}: *DLOAD {} på Euler-element stöds inte ännu (bara GRAV)", m.elem_name(dl.elem), dl.kind));
                    }
                }
            }
        }
        let mut nodes: Vec<usize> = elems.iter().flat_map(|el| el.nodes.iter().copied()).collect();
        nodes.sort_unstable();
        nodes.dedup();
        let ne = elems.len();
        Ok(Some(Euler { mats, elems, nodes, b1, b2, pending: Vec::new(), rates: vec![0.0; ne], initial_energy: 0.0, dt_est: None }))
    }

    /// Elementets tillstånd i analysens början ur *INITIAL CONDITIONS,
    /// TYPE=VOLUME FRACTION (raderna fylls nedifrån och upp, resten tomrum).
    fn init_states(&self, p: &Prepared) -> Result<Vec<Vec<f64>>> {
        let m = p.model;
        let mut fill: HashMap<usize, Vec<(String, f64)>> = HashMap::new();
        for kw in m.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(crate::inp::keywords::explicit_adv::VF_RESOLVED)) {
            for d in kw.data.iter().rev() {
                let e: usize = d.fields[0].parse().unwrap_or(usize::MAX);
                let v = crate::fem::model::parse_f64(&d.fields[1]).unwrap_or(0.0);
                fill.entry(e).or_default().push((d.fields[2].clone(), v));
            }
        }
        let em = super::sph::specific_energies(p)?;
        let mut out = Vec::with_capacity(self.elems.len());
        for el in &self.elems {
            let len: usize = el.mats.iter().map(|&i| stride(&self.mats[i])).sum();
            let mut st = vec![0.0; len];
            let mut total = 0.0;
            let mut off = 0;
            let mut offs = Vec::new();
            for &mi in &el.mats {
                offs.push(off);
                let mt = &self.mats[mi];
                mt.model.init_state(&mut st[off + MS..off + MS + mt.ns]);
                off += stride(mt);
            }
            if let Some(list) = fill.get(&el.e) {
                for (name, v) in list {
                    let Some(pos) = el.mats.iter().position(|&i| &self.mats[i].name == name) else {
                        return fail(format!("{}: materialinstansen '{name}' finns inte i elementets *EULERIAN SECTION", m.elem_name(el.e)));
                    };
                    let add = v.min(1.0 - total).max(0.0);
                    total += add;
                    st[offs[pos] + VF] += add;
                }
            }
            for (pos, &mi) in el.mats.iter().enumerate() {
                let mt = &self.mats[mi];
                let o = offs[pos];
                st[o + MASS] = st[o + VF] * el.vol0 * mt.rho0;
                if st[o + VF] > VF_TINY {
                    if let Some((p0, _)) = mt.model.init_energy(&mut st[o + MS..o + MS + mt.ns], em.get(&el.e).copied(), None, 0.0)? {
                        for k in 0..3 {
                            st[o + S + k] = -p0;
                        }
                    }
                }
            }
            out.push(st);
        }
        Ok(out)
    }

    fn offsets(&self, el: &Elem) -> Vec<usize> {
        let mut v = Vec::with_capacity(el.mats.len());
        let mut o = 0;
        for &mi in &el.mats {
            v.push(o);
            o += stride(&self.mats[mi]);
        }
        v
    }

    /// Elementets (index `k`) sammanlagda materialvolymandel i tillståndet `st`.
    pub fn material_fraction(&self, k: usize, st: &[f64]) -> f64 {
        let el = &self.elems[k];
        let offs = self.offsets(el);
        if st.is_empty() {
            return 0.0;
        }
        offs.iter().map(|&o| st.get(o + VF).copied().unwrap_or(0.0)).sum()
    }

    pub fn patch_mass(&self, _mass: &mut super::LumpedMass) {
        // massan sätts i varje kraftberäkning (den ändras med advektionen)
    }

    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink) -> Result<()> {
        let m = p.model;
        let ne = self.elems.len();
        let mut st: Vec<Vec<f64>> = Vec::with_capacity(ne);
        let init = if self.elems.iter().any(|el| ev.state_old[el.e].is_empty()) { Some(self.init_states(p)?) } else { None };
        for (k, el) in self.elems.iter().enumerate() {
            st.push(if ev.state_old[el.e].is_empty() { init.as_ref().unwrap()[k].clone() } else { ev.state_old[el.e].clone() });
        }
        if ev.time == 0.0 && ev.dt <= 0.0 {
            self.initial_energy = 0.0;
            let em = super::sph::specific_energies(p)?;
            for (k, el) in self.elems.iter().enumerate() {
                let offs = self.offsets(el);
                for (pos, _) in el.mats.iter().enumerate() {
                    self.initial_energy += st[k][offs[pos] + MASS] * em.get(&el.e).copied().unwrap_or(0.0);
                }
            }
        }
        self.pending.clear();
        if ev.dt > 0.0 {
            self.lagrange_and_advect(p, ev, &mut st)?;
        }
        // krafter, massa och stabilitetsgräns på det fasta nätet
        let mut nmass: HashMap<usize, f64> = HashMap::new();
        self.dt_est = None;
        for (k, el) in self.elems.iter().enumerate() {
            let offs = self.offsets(el);
            let mut sig = [0.0; 6];
            let mut mtot = 0.0;
            let mut cmax: f64 = 0.0;
            let mut vfm = 0.0;
            for (pos, &mi) in el.mats.iter().enumerate() {
                let o = offs[pos];
                let vf = st[k][o + VF];
                mtot += st[k][o + MASS];
                if vf > VF_TINY {
                    vfm += vf;
                    for c in 0..6 {
                        sig[c] += vf * st[k][o + S + c];
                    }
                    let mt = &self.mats[mi];
                    let rho = st[k][o + MASS] / (vf * el.vol0);
                    cmax = cmax.max(super::sph::sound_speed(mt.model.as_ref(), &st[k][o + MS..o + MS + mt.ns], rho));
                }
            }
            let s = super::sph::tensor(&sig);
            for (i, &n) in el.nodes.iter().enumerate() {
                for a in 0..3 {
                    let f: f64 = (0..3).map(|b| s[a][b] * el.b0[i][b]).sum();
                    out.fint[n][a] += f;
                    // tyngdkraften på materialets massa (yttre last, ALLWK)
                    out.fext[n][a] += mtot * el.gravity[a] / 8.0;
                }
                *nmass.entry(n).or_insert(0.0) += mtot / 8.0;
            }
            // volymviskositet
            let rate = self.rates[k];
            if vfm > VF_TINY && (self.b1 != 0.0 || self.b2 != 0.0) && ev.dt > 0.0 {
                let rho = mtot / el.vol0;
                let q = rho * el.le0 * (self.b1 * cmax * rate - if rate < 0.0 { self.b2 * self.b2 * el.le0 * rate * rate } else { 0.0 });
                for (i, &n) in el.nodes.iter().enumerate() {
                    for a in 0..3 {
                        out.fvis[n][a] += q * el.b0[i][a];
                    }
                }
            }
            if cmax > 0.0 {
                let vbar: [f64; 3] = std::array::from_fn(|c| el.nodes.iter().map(|&n| ev.v[n][c]).sum::<f64>() / 8.0);
                let sp = (vbar[0] * vbar[0] + vbar[1] * vbar[1] + vbar[2] * vbar[2]).sqrt();
                let dt = el.le0 / (cmax + sp);
                if self.dt_est.is_none_or(|b| dt < b.0) {
                    self.dt_est = Some((dt, el.e));
                }
            }
        }
        for &n in &self.nodes {
            let mn = nmass.get(&n).copied().unwrap_or(0.0);
            for c in 0..3 {
                out.mass.diag[n][c] = mn;
            }
        }
        for (k, el) in self.elems.iter().enumerate() {
            out.state_new[el.e] = std::mem::take(&mut st[k]);
        }
        let _ = m;
        Ok(())
    }

    /// Lagrangesteget och advektionen tillbaka till det fasta nätet.
    fn lagrange_and_advect(&mut self, p: &Prepared, ev: &Eval, st: &mut [Vec<f64>]) -> Result<()> {
        let m = p.model;
        let ne = self.elems.len();
        let dt = ev.dt;
        // Lagrangesteget per element
        let mut vlag = vec![0.0; ne];
        let mut vbar = vec![[0.0f64; 3]; ne];
        let mut mass_e = vec![0.0; ne];
        for k in 0..ne {
            let el = &self.elems[k];
            let xl: [[f64; 3]; 8] = std::array::from_fn(|i| std::array::from_fn(|c| m.nodes[el.nodes[i]][c] + ev.u[el.nodes[i]][c]));
            let xm: Vec<[f64; 3]> = (0..8).map(|i| std::array::from_fn(|c| m.nodes[el.nodes[i]][c] + 0.5 * ev.u[el.nodes[i]][c])).collect();
            vlag[k] = hex_volume(&xl);
            vbar[k] = std::array::from_fn(|c| el.nodes.iter().map(|&n| ev.v[n][c]).sum::<f64>() / 8.0);
            let Some(g) = super::super::dt::geometry(crate::elements::Family::Continuum3D, "C3D8R", &xm) else {
                return fail(format!("{}: Euler-elementet inverteras i Lagrangesteget (för långt tidsinkrement?)", m.elem_name(el.e)));
            };
            let mut l = [[0.0f64; 3]; 3];
            for i in 0..8 {
                for a in 0..3 {
                    for b in 0..3 {
                        l[a][b] += ev.v[el.nodes[i]][a] * g.b[i][b] / g.vol;
                    }
                }
            }
            let tr = l[0][0] + l[1][1] + l[2][2];
            self.rates[k] = tr;
            let d: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (l[a][b] + l[b][a]) * dt));
            let w: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (l[a][b] - l[b][a]) * dt));
            let deps = [d[0][0], d[1][1], d[2][2], 2.0 * d[0][1], 2.0 * d[0][2], 2.0 * d[1][2]];
            let jinc = (vlag[k] / el.vol0).max(1e-12);
            let offs = self.offsets(el);
            for (pos, &mi) in el.mats.iter().enumerate() {
                let o = offs[pos];
                mass_e[k] += st[k][o + MASS];
                if st[k][o + VF] <= VF_TINY {
                    continue;
                }
                let mt = &self.mats[mi];
                let rho_old = st[k][o + MASS] / (st[k][o + VF] * el.vol0);
                let rho_new = rho_old / jinc;
                let j = mt.rho0 / rho_new;
                let s0 = super::sph::tensor(&st[k][o + S..o + S + 6]);
                let ws: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|c| w[a][c] * s0[c][b] - s0[a][c] * w[c][b]).sum()));
                let srot = [s0[0][0] + ws[0][0], s0[1][1] + ws[1][1], s0[2][2] + ws[2][2], s0[0][1] + ws[0][1], s0[0][2] + ws[0][2], s0[1][2] + ws[1][2]];
                let jc = j.cbrt();
                let fdef = [[jc, 0.0, 0.0], [0.0, jc, 0.0], [0.0, 0.0, jc]];
                let ms_old: Vec<f64> = st[k][o + MS..o + MS + mt.ns].to_vec();
                let mut ms_new = vec![0.0; mt.ns];
                let input = MatInput { mode: StressMode::ThreeD, strain: &deps, dstrain: &deps, stress_old: &srot, time: ev.time, dt, temperature: 0.0, dtemp: 0.0, defgrad: Some(fdef) };
                let r = mt.model.update(&input, &ms_old, &mut ms_new).map_err(|err| FemError(format!("{} ({}): {}", m.elem_name(el.e), mt.name, err.0)))?;
                st[k][o + S..o + S + 6].copy_from_slice(&r.stress[..6]);
                st[k][o + MS..o + MS + mt.ns].copy_from_slice(&ms_new);
            }
        }
        // materialvolymer i den Lagrangeska cellen (samma töjning för alla material)
        let nm = self.mats.len();
        let mut vol = vec![vec![0.0f64; nm]; ne];
        let mut mas = vec![vec![0.0f64; nm]; ne];
        let mut sv = vec![vec![[0.0f64; 6]; nm]; ne];
        let mut sm: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); nm]; ne];
        let mut phi = vec![vec![0.0f64; nm]; ne];
        for k in 0..ne {
            let el = &self.elems[k];
            let offs = self.offsets(el);
            for (pos, &mi) in el.mats.iter().enumerate() {
                let o = offs[pos];
                let mt = &self.mats[mi];
                let vf = st[k][o + VF];
                phi[k][mi] = vf;
                vol[k][mi] = vf * vlag[k];
                mas[k][mi] = st[k][o + MASS];
                for c in 0..6 {
                    sv[k][mi][c] = st[k][o + S + c] * vol[k][mi];
                }
                sm[k][mi] = st[k][o + MS..o + MS + mt.ns].iter().map(|x| x * mas[k][mi]).collect();
            }
        }
        // ytornas svepta volymer och materialflöden
        struct Flux {
            from: usize,
            to: Option<usize>,
            dv: Vec<f64>,
            /// Givarens yta (för uppströmselementet över motstående yta).
            face: usize,
        }
        let mut fluxes: Vec<Flux> = Vec::new();
        for k in 0..ne {
            let el = &self.elems[k];
            for (f, fnodes) in crate::elements::continuum3d::HEX8_FACES.iter().enumerate() {
                let nb = el.nbr[f];
                if let Some((k2, _)) = nb {
                    if k2 < k {
                        continue;
                    }
                }
                let a: [[f64; 3]; 4] = std::array::from_fn(|i| m.nodes[el.nodes[fnodes[i]]]);
                let b: [[f64; 3]; 4] = std::array::from_fn(|i| std::array::from_fn(|c| a[i][c] + ev.u[el.nodes[fnodes[i]]][c]));
                // utåt från k: positivt när ytan flyttats utåt (k ger material till grannen)
                let s = swept(&a, &b);
                if s.abs() <= 1e-14 * el.vol0 {
                    continue;
                }
                let (donor, acceptor, dface) = if s > 0.0 {
                    (k, nb.map(|x| x.0), f)
                } else {
                    match nb {
                        Some((k2, f2)) => (k2, Some(k), f2),
                        // inflöde genom en fri rand: tomrum
                        None => continue,
                    }
                };
                let sa = s.abs();
                let de = &self.elems[donor];
                let mut dv = vec![0.0; nm];
                let mut tot = 0.0;
                for &mi in &de.mats {
                    let ph = phi[donor][mi];
                    let vm = vol[donor][mi];
                    // gränsytan i ett blandat element (givare–mottagare, Hirt & Nichols):
                    // materialet ligger mot ytan när mottagaren är fylligare än
                    // uppströmselementet (motstående yta) och flödar då först,
                    // annars flödar först det övriga innehållet
                    let x = if ph > VF_TINY && ph < 1.0 - VF_TINY {
                        let up = de.nbr[OPPOSITE[dface]].map_or(0.0, |x| phi[x.0][mi]);
                        let dn = acceptor.map_or(0.0, |a| phi[a][mi]);
                        if dn >= up {
                            sa.min(vm)
                        } else {
                            (sa - (vlag[donor] - vm)).clamp(0.0, vm)
                        }
                    } else {
                        (sa * ph.clamp(0.0, 1.0)).min(vm)
                    };
                    dv[mi] = x;
                    tot += x;
                }
                if tot > sa {
                    for x in dv.iter_mut() {
                        *x *= sa / tot;
                    }
                }
                fluxes.push(Flux { from: donor, to: acceptor, dv, face: dface });
            }
        }
        // inget material får bli negativt: skala givarens flöden
        let mut out_req = vec![vec![0.0f64; nm]; ne];
        for fl in &fluxes {
            for mi in 0..nm {
                out_req[fl.from][mi] += fl.dv[mi];
            }
        }
        let scale: Vec<Vec<f64>> = (0..ne).map(|k| (0..nm).map(|mi| if out_req[k][mi] > vol[k][mi] && out_req[k][mi] > 0.0 { vol[k][mi] / out_req[k][mi] } else { 1.0 }).collect()).collect();
        // transport (värden i givaren före advektionen)
        let mut nvol = vol.clone();
        let mut nmas = mas.clone();
        let mut nsv = sv.clone();
        let mut nsm = sm.clone();
        let mut mom: Vec<[f64; 3]> = (0..ne).map(|k| std::array::from_fn(|c| mass_e[k] * vbar[k][c])).collect();
        // andra ordningen: begränsad lutning (minmod) mellan uppströms-, givar- och
        // mottagarelementet för täthet, spänning, tillstånd och hastighet
        let lim = |qd: f64, qa: f64, qu: f64, courant: f64| qd + 0.5 * (1.0 - courant) * minmod(qa - qd, qd - qu);
        for fl in &fluxes {
            let d = fl.from;
            let up = self.elems[d].nbr[OPPOSITE[fl.face]].map(|x| x.0);
            let second = self.elems[d].second_order;
            for mi in 0..nm {
                let x = fl.dv[mi] * scale[d][mi];
                if x <= 0.0 || vol[d][mi] <= 0.0 {
                    continue;
                }
                let frac = x / vol[d][mi];
                // värden på ytan: givarens (första ordningen) eller begränsade
                let three = match (second, fl.to, up) {
                    (true, Some(a), Some(u)) if vol[a][mi] > 0.0 && vol[u][mi] > 0.0 && mas[a][mi] > 0.0 && mas[u][mi] > 0.0 => Some((a, u)),
                    _ => None,
                };
                let courant = frac.min(1.0);
                let rho_d = mas[d][mi] / vol[d][mi];
                let (dm, sig_f, st_f, vel_f): (f64, [f64; 6], Vec<f64>, [f64; 3]) = match three {
                    Some((a, u)) => {
                        let rho = lim(rho_d, mas[a][mi] / vol[a][mi], mas[u][mi] / vol[u][mi], courant).max(0.0);
                        let sig: [f64; 6] = std::array::from_fn(|c| lim(sv[d][mi][c] / vol[d][mi], sv[a][mi][c] / vol[a][mi], sv[u][mi][c] / vol[u][mi], courant));
                        let ns = sm[d][mi].len();
                        let st: Vec<f64> = (0..ns).map(|c| if sm[a][mi].len() == ns && sm[u][mi].len() == ns && mas[d][mi] > 0.0 { lim(sm[d][mi][c] / mas[d][mi], sm[a][mi][c] / mas[a][mi], sm[u][mi][c] / mas[u][mi], courant) } else { sm[d][mi][c] / mas[d][mi].max(1e-300) }).collect();
                        let vel: [f64; 3] = std::array::from_fn(|c| lim(vbar[d][c], vbar[a][c], vbar[u][c], courant));
                        ((x * rho).min(mas[d][mi]), sig, st, vel)
                    }
                    None => {
                        let sig: [f64; 6] = std::array::from_fn(|c| sv[d][mi][c] / vol[d][mi]);
                        let st: Vec<f64> = sm[d][mi].iter().map(|q| q / mas[d][mi].max(1e-300)).collect();
                        (mas[d][mi] * frac, sig, st, vbar[d])
                    }
                };
                let _ = rho_d;
                nvol[d][mi] -= x;
                nmas[d][mi] -= dm;
                for c in 0..6 {
                    let q = sig_f[c] * x;
                    nsv[d][mi][c] -= q;
                    if let Some(a) = fl.to {
                        nsv[a][mi][c] += q;
                    }
                }
                let ns = sm[d][mi].len();
                for c in 0..ns {
                    let q = st_f[c] * dm;
                    nsm[d][mi][c] -= q;
                    if let Some(a) = fl.to {
                        if nsm[a][mi].len() == ns {
                            nsm[a][mi][c] += q;
                        }
                    }
                }
                for c in 0..3 {
                    mom[d][c] -= dm * vel_f[c];
                }
                if let Some(a) = fl.to {
                    nvol[a][mi] += x;
                    nmas[a][mi] += dm;
                    for c in 0..3 {
                        mom[a][c] += dm * vel_f[c];
                    }
                }
            }
        }
        // nytt tillstånd på det fasta nätet
        let mut dvel = vec![[0.0f64; 3]; ne];
        let mut mnew = vec![0.0f64; ne];
        for k in 0..ne {
            let el = &self.elems[k];
            let offs = self.offsets(el);
            let tot: f64 = el.mats.iter().map(|&mi| nvol[k][mi].max(0.0)).sum::<f64>() / el.vol0;
            // överfyllt element (kompressibilitet): volymandelarna skalas till 1
            let fac = if tot > 1.0 { 1.0 / tot } else { 1.0 };
            for (pos, &mi) in el.mats.iter().enumerate() {
                let o = offs[pos];
                let mt = &self.mats[mi];
                let v = nvol[k][mi].max(0.0);
                let ms = nmas[k][mi].max(0.0);
                let vf = v / el.vol0 * fac;
                if vf <= VF_TINY || ms <= 0.0 {
                    st[k][o + VF] = 0.0;
                    st[k][o + MASS] = 0.0;
                    for c in 0..6 {
                        st[k][o + S + c] = 0.0;
                    }
                    mt.model.init_state(&mut st[k][o + MS..o + MS + mt.ns]);
                    continue;
                }
                st[k][o + VF] = vf;
                st[k][o + MASS] = ms;
                mnew[k] += ms;
                for c in 0..6 {
                    st[k][o + S + c] = nsv[k][mi][c] / v;
                }
                if nsm[k][mi].len() == mt.ns {
                    for c in 0..mt.ns {
                        st[k][o + MS + c] = nsm[k][mi][c] / ms;
                    }
                }
                let rho = ms / (vf * el.vol0);
                crate::materials::eos::set_jacobian(mt.model.as_ref(), &mut st[k][o + MS..o + MS + mt.ns], mt.rho0 / rho);
            }
            if mnew[k] > 0.0 {
                for c in 0..3 {
                    dvel[k][c] = mom[k][c] / mnew[k] - vbar[k][c];
                }
            }
        }
        // nodernas hastigheter (elementmittprojektion)
        let mut num: HashMap<usize, [f64; 4]> = HashMap::new();
        for k in 0..ne {
            for &n in &self.elems[k].nodes {
                let e = num.entry(n).or_insert([0.0; 4]);
                for c in 0..3 {
                    e[c] += mnew[k] * dvel[k][c];
                }
                e[3] += mnew[k];
            }
        }
        for &n in &self.nodes {
            let e = num.get(&n).copied().unwrap_or([0.0; 4]);
            let v: [f64; 3] = if e[3] > 0.0 { std::array::from_fn(|c| ev.v[n][c] + e[c] / e[3]) } else { [0.0; 3] };
            self.pending.push((n, v));
        }
        Ok(())
    }

    /// Euler-noderna tillbaka till sina lägen, hastigheterna efter advektionen
    /// (utom låsta frihetsgrader).
    pub fn after_forces(&mut self, u: &mut [NodeVec], vh: &mut [NodeVec], fixed: &[u16]) {
        for &n in &self.nodes {
            for c in 0..3 {
                if fixed[n] & (1 << c) == 0 {
                    u[n][c] = 0.0;
                }
            }
        }
        for &(n, v) in &self.pending {
            for c in 0..3 {
                if fixed[n] & (1 << c) == 0 {
                    vh[n][c] = v[c];
                }
            }
        }
    }

    pub fn output(&self, p: &Prepared, fr: &mut FrameData, state: &[Vec<f64>]) -> Result<()> {
        let init = if self.elems.iter().any(|el| state[el.e].is_empty()) { Some(self.init_states(p)?) } else { None };
        for (k, el) in self.elems.iter().enumerate() {
            let st = if state[el.e].is_empty() { &init.as_ref().unwrap()[k] } else { &state[el.e] };
            let offs = self.offsets(el);
            let mut fields = Vec::new();
            let mut tot = 0.0;
            let mut sig = [0.0; 6];
            let mut peeq = 0.0;
            let mut has_peeq = false;
            let mut dens = 0.0;
            for (pos, &mi) in el.mats.iter().enumerate() {
                let o = offs[pos];
                let mt = &self.mats[mi];
                let vf = st[o + VF];
                let nm = format!("EVF_{}", mt.name);
                let mut f = IpField::new(&nm, &[&nm]);
                f.points.push((1, 0, vec![vf]));
                fields.push(f);
                tot += vf;
                for c in 0..6 {
                    sig[c] += vf * st[o + S + c];
                }
                if vf > VF_TINY {
                    dens += st[o + MASS] / el.vol0;
                }
                let names = mt.model.output_names();
                if let Some(ip) = names.iter().position(|n| *n == "PEEQ") {
                    has_peeq = true;
                    peeq += vf * mt.model.output_values(&st[o + MS..o + MS + mt.ns]).get(ip).copied().unwrap_or(0.0);
                }
            }
            let mut f = IpField::new("EVF_VOID", &["EVF_VOID"]);
            f.points.push((1, 0, vec![(1.0 - tot).max(0.0)]));
            fields.push(f);
            let w = if tot > VF_TINY { 1.0 / tot } else { 0.0 };
            let mut s = IpField::new("SVAVG", &["SVAVG11", "SVAVG22", "SVAVG33", "SVAVG12", "SVAVG13", "SVAVG23"]);
            s.points.push((1, 0, sig.iter().map(|x| x * w).collect()));
            fields.push(s);
            if has_peeq {
                let mut f = IpField::new("PEEQVAVG", &["PEEQVAVG"]);
                f.points.push((1, 0, vec![peeq * w]));
                fields.push(f);
            }
            let mut d = IpField::new("DENSITYVAVG", &["DENSITYVAVG"]);
            d.points.push((1, 0, vec![dens * w]));
            fields.push(d);
            if let Some(slot) = fr.elem.get_mut(el.e) {
                *slot = fields;
            }
        }
        Ok(())
    }
}
