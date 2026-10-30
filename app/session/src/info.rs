//! Modellträdet som gränssnittet visar, och symboler för randvillkor och laster.

use rustyfem::fem::dof;
use rustyfem::fem::model::{BcBlock, CloadBlock, FeModel, RawKeyword, SectionKind};
use serde::Serialize;
use specta::Type;
use std::collections::BTreeMap;

/// Högsta antal rader per block som skickas till gränssnittet.
const MAX_ITEMS: usize = 200;

#[derive(Debug, Clone, Serialize, Type)]
pub struct SetInfo {
    pub name: String,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct KeywordInfo {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub lines: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct MaterialInfo {
    pub index: u32,
    pub name: String,
    pub density: f64,
    pub behaviors: Vec<KeywordInfo>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct SectionInfo {
    pub index: u32,
    pub elset: String,
    pub material: String,
    pub kind: String,
    pub thickness: Option<f64>,
    pub shape: Option<String>,
    pub elements: u32,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct ItemInfo {
    /// Nod eller element ("INSTANS.nummer").
    pub target: String,
    /// Frihetsgrad (U1, UR3, ...) eller lasttyp (P, GRAV, ...).
    pub what: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct BlockInfo {
    pub index: u32,
    pub kind: String,
    pub op_new: bool,
    pub amplitude: Option<String>,
    pub source: String,
    pub total: u32,
    pub items: Vec<ItemInfo>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct StepInfo {
    pub index: u32,
    pub name: String,
    pub procedure: String,
    pub nlgeom: Option<bool>,
    pub perturbation: bool,
    pub max_increments: u32,
    pub procedure_keyword: Option<KeywordInfo>,
    pub bcs: Vec<BlockInfo>,
    pub cloads: Vec<BlockInfo>,
    pub dloads: Vec<BlockInfo>,
    pub other: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct ModelInfo {
    pub heading: String,
    pub dim: u32,
    pub nodes: u32,
    pub elements: u32,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub instances: Vec<String>,
    pub element_types: Vec<SetInfo>,
    pub nsets: Vec<SetInfo>,
    pub elsets: Vec<SetInfo>,
    pub surfaces: Vec<SetInfo>,
    pub materials: Vec<MaterialInfo>,
    pub sections: Vec<SectionInfo>,
    pub initial_bcs: Vec<BlockInfo>,
    pub steps: Vec<StepInfo>,
    pub other: Vec<String>,
    pub warnings: Vec<String>,
}

fn keyword(k: &RawKeyword) -> KeywordInfo {
    KeywordInfo { name: k.name.clone(), params: k.params.clone(), lines: k.data.iter().map(|d| d.fields.clone()).collect() }
}

fn bc_block(m: &FeModel, i: usize, b: &BcBlock) -> BlockInfo {
    BlockInfo {
        index: i as u32,
        kind: if b.kind.is_empty() { "DISPLACEMENT".into() } else { b.kind.clone() },
        op_new: b.op_new,
        amplitude: b.amplitude.clone(),
        source: b.source.clone(),
        total: b.items.len() as u32,
        items: b.items.iter().take(MAX_ITEMS).map(|&(n, d, v)| ItemInfo { target: m.node_name(n), what: dof::name(d).into(), value: v }).collect(),
    }
}

fn cload_block(m: &FeModel, i: usize, b: &CloadBlock) -> BlockInfo {
    BlockInfo {
        index: i as u32,
        kind: if b.follower { "FOLLOWER".into() } else { "CLOAD".into() },
        op_new: b.op_new,
        amplitude: b.amplitude.clone(),
        source: b.source.clone(),
        total: b.items.len() as u32,
        items: b.items.iter().take(MAX_ITEMS).map(|&(n, d, v)| ItemInfo { target: m.node_name(n), what: dof::name(d).into(), value: v }).collect(),
    }
}

/// Modellträdet.
pub fn model_info(m: &FeModel) -> ModelInfo {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &m.nodes {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if m.nodes.is_empty() {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }
    let mut types: BTreeMap<&str, u32> = BTreeMap::new();
    for e in &m.elements {
        *types.entry(e.etype.name()).or_default() += 1;
    }
    let sets = |s: &BTreeMap<String, Vec<usize>>| s.iter().map(|(k, v)| SetInfo { name: k.clone(), count: v.len() as u32 }).collect();
    ModelInfo {
        heading: m.heading.trim().to_string(),
        dim: m.dim() as u32,
        nodes: m.nodes.len() as u32,
        elements: m.elements.len() as u32,
        bbox_min: lo,
        bbox_max: hi,
        instances: m.instances.clone(),
        element_types: types.into_iter().map(|(k, v)| SetInfo { name: k.into(), count: v }).collect(),
        nsets: sets(&m.nsets),
        elsets: sets(&m.elsets),
        surfaces: m.surfaces.iter().map(|(k, s)| SetInfo { name: k.clone(), count: (s.faces.len() + s.nodes.len()) as u32 }).collect(),
        materials: m
            .materials
            .iter()
            .enumerate()
            .map(|(i, mt)| MaterialInfo { index: i as u32, name: mt.name.clone(), density: mt.density, behaviors: mt.behaviors.iter().map(keyword).collect() })
            .collect(),
        sections: m
            .sections
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (kind, thickness, shape) = match &s.kind {
                    SectionKind::Solid { thickness } => ("SOLID", *thickness, None),
                    SectionKind::Shell { thickness, .. } => ("SHELL", Some(*thickness), None),
                    SectionKind::Beam(b) => ("BEAM", None, Some(b.shape.clone())),
                    SectionKind::Other => ("OTHER", None, None),
                };
                SectionInfo {
                    index: i as u32,
                    elset: s.elset.clone(),
                    material: s.material.clone(),
                    kind: if matches!(s.kind, SectionKind::Other) { s.raw.name.clone() } else { kind.into() },
                    thickness,
                    shape,
                    elements: m.elsets.get(&s.elset).map(|v| v.len() as u32).unwrap_or(0),
                }
            })
            .collect(),
        initial_bcs: m.initial_bcs.iter().enumerate().map(|(i, b)| bc_block(m, i, b)).collect(),
        steps: m
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| StepInfo {
                index: i as u32,
                name: s.name.clone(),
                procedure: s.procedure_name().to_string(),
                nlgeom: s.nlgeom,
                perturbation: s.perturbation,
                max_increments: s.max_increments as u32,
                procedure_keyword: s.procedure.as_ref().map(keyword),
                bcs: s.bcs.iter().enumerate().map(|(k, b)| bc_block(m, k, b)).collect(),
                cloads: s.cloads.iter().enumerate().map(|(k, b)| cload_block(m, k, b)).collect(),
                dloads: s
                    .dloads
                    .iter()
                    .enumerate()
                    .map(|(k, b)| BlockInfo {
                        index: k as u32,
                        kind: "DLOAD".into(),
                        op_new: b.op_new,
                        amplitude: b.amplitude.clone(),
                        source: b.source.clone(),
                        total: b.items.len() as u32,
                        items: b.items.iter().take(MAX_ITEMS).map(|d| ItemInfo { target: m.elem_name(d.elem), what: d.kind.clone(), value: d.magnitude }).collect(),
                    })
                    .collect(),
                other: s.extra.iter().map(|k| k.name.clone()).collect(),
            })
            .collect(),
        other: m.extra.iter().map(|k| k.name.clone()).collect(),
        warnings: m.warnings.clone(),
    }
}

/// Randvillkor och punktlaster som gäller i ett steg (ärvda enligt OP=NEW),
/// packade som `u32` antal randvillkorsnoder och lastnoder, därefter
/// (nod u32, slotmask u32) per randvillkor och (nod u32, F1–F3 f32, M1–M3 f32)
/// per last.
pub fn symbols(m: &FeModel, step: Option<usize>) -> Vec<u8> {
    let mut bc: BTreeMap<usize, u32> = BTreeMap::new();
    let mut load: BTreeMap<usize, [f64; 6]> = BTreeMap::new();
    let add_bcs = |bc: &mut BTreeMap<usize, u32>, blocks: &[BcBlock]| {
        for b in blocks {
            if b.op_new {
                bc.clear();
            }
            for &(n, d, _) in &b.items {
                if let Some(s) = dof::slot(d) {
                    *bc.entry(n).or_default() |= 1 << s;
                }
            }
        }
    };
    add_bcs(&mut bc, &m.initial_bcs);
    if let Some(si) = step {
        for st in m.steps.iter().take(si + 1) {
            add_bcs(&mut bc, &st.bcs);
            let mut cl: BTreeMap<usize, [f64; 6]> = if st.perturbation { BTreeMap::new() } else { load.clone() };
            for b in &st.cloads {
                if b.op_new {
                    cl.clear();
                }
                for &(n, d, v) in &b.items {
                    if let Some(s) = dof::slot(d).filter(|&s| s < 6) {
                        cl.entry(n).or_default()[s] += v;
                    }
                }
            }
            load = cl;
        }
    }
    let mut b = Vec::with_capacity(8 + bc.len() * 8 + load.len() * 28);
    b.extend((bc.len() as u32).to_le_bytes());
    b.extend((load.len() as u32).to_le_bytes());
    for (n, mask) in &bc {
        b.extend((*n as u32).to_le_bytes());
        b.extend(mask.to_le_bytes());
    }
    for (n, f) in &load {
        b.extend((*n as u32).to_le_bytes());
        for v in f {
            b.extend((*v as f32).to_le_bytes());
        }
    }
    b
}
