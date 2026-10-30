//! Nyckelord för värmeledning och termisk spänning.
//!
//! - *HEAT TRANSFER: procedur (stationär eller transient), se `steps/heat.rs`.
//! - *COUPLED TEMPERATURE-DISPLACEMENT: procedur, se `steps/heat_coupled.rs`.
//! - *CONDUCTIVITY, *SPECIFIC HEAT: materialbeteenden (`materials/heat.rs`).
//!   *LATENT HEAT (fasomvandling, entalpiform i kapacitetstermen). *EXPANSION registreras i
//!   `material_behaviors.rs` (`materials/thermal.rs`).
//! - *CFLUX: koncentrerat värmeflöde, lagras som punktlast på frihetsgrad 11
//!   så att stegens arvsregler (OP=MOD/NEW, ramp) gäller som för *CLOAD.
//! - *DFLUX, *FILM, *RADIATE (och ytvarianterna *DSFLUX, *SFILM, *SRADIATE):
//!   lagras som fördelade laster med typerna `DFLUX:BF`, `DFLUX:Sn`,
//!   `FILM:Fn` (storlek h), `FILMS:Fn` (storlek h·T∞) och `RAD:Rn` (storlek ε,
//!   T∞ som extra värde). Ytor (`Sn`) och film/strålning (`Fn`, `Rn`) har samma
//!   nummer som elementets ytor.
//! - *PHYSICAL CONSTANTS och *FILM PROPERTY (h(T), används med namnet i
//!   stället för h på *FILM-raden): modelldata, sparas i `FeModel::extra`.
//! - AMPLITUDE= på *CFLUX, *DFLUX, *DSFLUX, *RADIATE och *FILM (T∞) samt
//!   FILM AMPLITUDE= (h) utvärderas av `steps/heat.rs`.
//! - *INITIAL CONDITIONS, TYPE=TEMPERATURE och *TEMPERATURE (även FILE=jobb)
//!   registreras i `temperature.rs`; värmeledningen läser starttemperaturerna
//!   därifrån (`steps/heat.rs`).

use crate::fem::dof::NT11;
use crate::fem::model::{parse_f64, CloadBlock, DistLoad, DloadBlock, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{parse_label, procedure, store_model, InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CFLUX", kind: Kind::Step, handler: cflux },
    Keyword { name: "CONDUCTIVITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "COUPLED TEMPERATURE-DISPLACEMENT", kind: Kind::Step, handler: procedure },
    Keyword { name: "DFLUX", kind: Kind::Step, handler: dflux },
    Keyword { name: "DSFLUX", kind: Kind::Step, handler: dsflux },
    Keyword { name: "FILM", kind: Kind::Step, handler: film },
    Keyword { name: "FILM PROPERTY", kind: Kind::Model, handler: store_model },
    Keyword { name: "HEAT TRANSFER", kind: Kind::Step, handler: procedure },
    Keyword { name: "LATENT HEAT", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "PHYSICAL CONSTANTS", kind: Kind::Model, handler: store_model },
    Keyword { name: "RADIATE", kind: Kind::Step, handler: radiate },
    Keyword { name: "SFILM", kind: Kind::Step, handler: sfilm },
    Keyword { name: "SPECIFIC HEAT", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SRADIATE", kind: Kind::Step, handler: sradiate },
];

fn num(kw: &RawKeyword, line: usize, s: &str) -> Result<f64> {
    parse_f64(s).ok_or_else(|| FemError(format!("{} rad {line} (*{}): '{s}' är inget tal", kw.file, kw.name)))
}

fn op_new(kw: &RawKeyword) -> bool {
    kw.param_upper("OP").is_some_and(|o| o == "NEW")
}

fn no_amplitude(kw: &RawKeyword) -> Result<()> {
    for p in ["USER", "FIELD"] {
        if kw.has(p) {
            return kw.err(format!("{p}= stöds inte ännu"));
        }
    }
    Ok(())
}

/// AMPLITUDE= (namnet med versaler).
fn amplitude(kw: &RawKeyword) -> Option<String> {
    kw.param("AMPLITUDE").map(|s| s.to_uppercase())
}

fn cflux(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = CloadBlock { op_new: op_new(kw), amplitude: amplitude(kw), follower: false, items: Vec::new(), source: kw.location() };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'nod/nset, 11, flöde'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        let k = parse_label(&d.fields[1]).unwrap_or(0);
        if k != 11 {
            return kw.err_at(d.line, format!("frihetsgrad {} – *CFLUX gäller temperaturen (11)", d.fields[1].trim()));
        }
        let v = num(kw, d.line, &d.fields[2])?;
        for n in nodes {
            block.items.push((n, NT11, v));
        }
    }
    ctx.step_mut(kw)?.cloads.push(block);
    Ok(())
}

fn dblock(kw: &RawKeyword) -> DloadBlock {
    DloadBlock { op_new: op_new(kw), amplitude: amplitude(kw), items: Vec::new(), source: kw.location() }
}

/// "S3" → 3, "F12" → None om det inte är `prefix` + nummer.
fn face_no(label: &str, prefix: char) -> Option<usize> {
    label.strip_prefix(prefix).and_then(|s| s.parse::<usize>().ok()).filter(|&n| n >= 1)
}

fn dflux(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'element/elset, BF eller Sn, flöde'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        if label != "BF" && face_no(&label, 'S').is_none() {
            return kw.err_at(d.line, format!("flödestypen {label} stöds inte (BF, S1, S2, ...)"));
        }
        let q = num(kw, d.line, &d.fields[2])?;
        for e in elems {
            block.items.push(DistLoad { elem: e, kind: format!("DFLUX:{label}"), magnitude: q, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

fn push_film(block: &mut DloadBlock, e: usize, face: usize, sink: f64, h: f64) {
    block.items.push(DistLoad { elem: e, kind: format!("FILM:F{face}"), magnitude: h, extra: Vec::new() });
    block.items.push(DistLoad { elem: e, kind: format!("FILMS:F{face}"), magnitude: h * sink, extra: Vec::new() });
}

/// Filmlaster: h (FILM) får FILM AMPLITUDE=, h·T∞ (FILMS) AMPLITUDE= eller
/// FILM AMPLITUDE=. En filmegenskap (*FILM PROPERTY) i stället för h ger
/// `FILMP:Fn:NAMN` med T∞ som extra värde (AMPLITUDE= stöds då inte).
fn film_blocks(ctx: &mut InpCtx, kw: &RawKeyword, block: DloadBlock) -> Result<()> {
    let a_sink = amplitude(kw);
    let a_h = kw.param("FILM AMPLITUDE").map(|s| s.to_uppercase());
    if a_sink.is_some() && a_h.is_some() {
        return kw.err("AMPLITUDE= och FILM AMPLITUDE= samtidigt stöds inte ännu");
    }
    if a_sink.is_none() && a_h.is_none() {
        ctx.step_mut(kw)?.dloads.push(block);
        return Ok(());
    }
    if block.items.iter().any(|l| l.kind.starts_with("FILMP:")) {
        return kw.err("amplitud tillsammans med *FILM PROPERTY stöds inte ännu");
    }
    let mut hb = DloadBlock { op_new: block.op_new, amplitude: a_h.clone(), items: Vec::new(), source: block.source.clone() };
    let mut sb = DloadBlock { op_new: false, amplitude: a_h.or(a_sink), items: Vec::new(), source: block.source.clone() };
    for l in block.items {
        if l.kind.starts_with("FILMS:") {
            sb.items.push(l);
        } else {
            hb.items.push(l);
        }
    }
    let st = ctx.step_mut(kw)?;
    st.dloads.push(hb);
    st.dloads.push(sb);
    Ok(())
}

fn film(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        // FnNU: filmkoefficient och omgivningstemperatur från FILM (user/loads.rs)
        if let Some(face) = d.fields.get(1).map(|l| l.trim().to_uppercase()).and_then(|l| l.strip_suffix("NU").and_then(|f| face_no(f, 'F'))) {
            let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
            let set = if crate::inp::parse_label(&d.fields[0]).is_some() { String::new() } else { d.fields[0].trim().to_uppercase() };
            for e in elems {
                block.items.push(DistLoad { elem: e, kind: format!("FILMU:F{face}:{set}"), magnitude: 1.0, extra: Vec::new() });
            }
            continue;
        }
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "skriv 'element/elset, Fn, omgivningstemperatur, filmkoefficient'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(face) = face_no(&label, 'F') else {
            return kw.err_at(d.line, format!("filmtypen {label} stöds inte (F1, F2, ...)"));
        };
        let sink = num(kw, d.line, &d.fields[2])?;
        let hs = d.fields[3].trim();
        if parse_f64(hs).is_none() && !hs.is_empty() {
            // filmegenskap (*FILM PROPERTY) i stället för filmkoefficient
            for e in elems {
                block.items.push(DistLoad { elem: e, kind: format!("FILMP:F{face}:{}", hs.to_uppercase()), magnitude: 1.0, extra: vec![sink] });
            }
            continue;
        }
        let h = num(kw, d.line, hs)?;
        for e in elems {
            push_film(&mut block, e, face, sink, h);
        }
    }
    film_blocks(ctx, kw, block)
}

fn radiate(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "skriv 'element/elset, Rn, omgivningstemperatur, emissivitet'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(face) = face_no(&label, 'R') else {
            return kw.err_at(d.line, format!("strålningstypen {label} stöds inte (R1, R2, ...)"));
        };
        let sink = num(kw, d.line, &d.fields[2])?;
        let eps = num(kw, d.line, &d.fields[3])?;
        for e in elems {
            block.items.push(DistLoad { elem: e, kind: format!("RAD:R{face}"), magnitude: eps, extra: vec![sink] });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

/// Elementytor (element, ytnummer) för en *SURFACE.
fn surface_faces(ctx: &InpCtx, kw: &RawKeyword, line: usize, name: &str) -> Result<Vec<(usize, usize)>> {
    let up = name.trim().to_uppercase();
    let mesh = ctx.mesh_ref();
    let Some(s) = mesh.surfaces.get(&up) else {
        return kw.err_at(line, format!("ytan '{}' finns inte", name.trim()));
    };
    if s.kind != "ELEMENT" {
        return kw.err_at(line, format!("ytan '{}' är inte en elementyta", name.trim()));
    }
    let mut out = Vec::new();
    for (e, f) in &s.faces {
        match face_no(f, 'S') {
            Some(n) => out.push((*e, n)),
            None => return kw.err_at(line, format!("ytbeteckningen {f} stöds inte för värmeflöden (S1, S2, ...)")),
        }
    }
    Ok(out)
}

fn dsflux(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'yta, S, flöde'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "S" {
            return kw.err_at(d.line, format!("flödestypen {label} stöds inte (S)"));
        }
        let q = num(kw, d.line, &d.fields[2])?;
        for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
            block.items.push(DistLoad { elem: e, kind: format!("DFLUX:S{f}"), magnitude: q, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

fn sfilm(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "skriv 'yta, F, omgivningstemperatur, filmkoefficient'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "F" {
            return kw.err_at(d.line, format!("filmtypen {label} stöds inte (F)"));
        }
        let sink = num(kw, d.line, &d.fields[2])?;
        let hs = d.fields[3].trim();
        if parse_f64(hs).is_none() && !hs.is_empty() {
            for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
                block.items.push(DistLoad { elem: e, kind: format!("FILMP:F{f}:{}", hs.to_uppercase()), magnitude: 1.0, extra: vec![sink] });
            }
            continue;
        }
        let h = num(kw, d.line, hs)?;
        for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
            push_film(&mut block, e, f, sink, h);
        }
    }
    film_blocks(ctx, kw, block)
}

fn sradiate(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "skriv 'yta, R, omgivningstemperatur, emissivitet'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "R" {
            return kw.err_at(d.line, format!("strålningstypen {label} stöds inte (R)"));
        }
        let sink = num(kw, d.line, &d.fields[2])?;
        let eps = num(kw, d.line, &d.fields[3])?;
        for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
            block.items.push(DistLoad { elem: e, kind: format!("RAD:R{f}"), magnitude: eps, extra: vec![sink] });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}
