//! Regressionstester: kör en .inp i RustyFEM och jämför med en sparad
//! baslinje (`fall.ref`) som RustyFEM själv har skrivit.
//!
//! - `case`: fall under `tests/cases/<område>/` (i testmappen `RustyFEM tests`)
//!   med sidofil och baslinje.
//! - `compare`: jämförelse per storhet med tolerans.
//!
//! `rustyfem baseline fall.inp` skriver om `fall.ref` från RustyFEM;
//! `rustyfem check fall.inp` jämför mot den. `cargo test` i testmappen
//! jämför alla fall mot sina baslinjer (tests/regression.rs).

pub mod case;
pub mod compare;

use crate::fem::output::ResultSet;
use crate::model::{FemError, Result};
use crate::steps::{self, RunOptions};
use std::path::{Path, PathBuf};

pub use case::CaseSpec;
pub use compare::CaseReport;

/// Kör en .inp-fil i RustyFEM och returnerar resultaten.
pub fn run_rustyfem(inp: &Path, all_frames: bool) -> Result<(ResultSet, crate::inp::InpModel)> {
    let m = crate::inp::read_file(inp)?;
    let (out, _) = steps::run(&m.model, &RunOptions { all_frames, ..Default::default() })?;
    Ok((out.to_result_set(&m.model), m))
}

/// Kör ett fall i RustyFEM med sidofilens tidigare jobb (`global_inp`,
/// `old_inp`), som körs först i en temporär jobbkatalog.
pub fn run_rustyfem_spec(inp: &Path, spec: &CaseSpec) -> Result<(ResultSet, crate::inp::InpModel)> {
    if spec.global_inp.is_none() && spec.old_inp.is_none() {
        return run_rustyfem(inp, spec.all_frames);
    }
    use crate::steps::restart::{read_job, scope, Job};
    let dir = unique_dir(&format!("rf-{}", case::case_name(inp)))?;
    let base = inp.parent().unwrap_or(Path::new("."));
    let prior = |file: &str| -> Result<(std::path::PathBuf, String)> {
        let p = base.join(file);
        let mut j = Job::for_file(&p);
        j.dir = dir.clone();
        let stem = j.name.clone();
        let m = read_job(&p, &mut j)?;
        let _s = scope(j);
        let (out, _) = steps::run(&m.model, &RunOptions::default()).map_err(|e| FemError(format!("{}: {}", p.display(), e.0)))?;
        let res = dir.join(format!("{stem}.res"));
        let text = out.to_result_set(&m.model).to_text() + &crate::steps::submodel::mesh_text(&m.model);
        std::fs::write(&res, text).map_err(|e| FemError(format!("kunde inte skriva {}: {e}", res.display())))?;
        Ok((res, stem))
    };
    let mut job = Job::for_file(inp);
    job.dir = dir.clone();
    if let Some(g) = &spec.global_inp {
        job.global = Some(prior(g)?.0);
    }
    if let Some(o) = &spec.old_inp {
        let (_, stem) = prior(o)?;
        job.oldjob = Some(dir.join(stem));
    }
    let res = (|| {
        let m = read_job(inp, &mut job)?;
        let _s = scope(job.clone());
        let (out, _) = steps::run(&m.model, &RunOptions { all_frames: spec.all_frames, ..Default::default() })?;
        Ok((out.to_result_set(&m.model), m))
    })();
    if !crate::settings::with(|s| s.keep_work_dirs) {
        let _ = std::fs::remove_dir_all(&dir);
    }
    res
}

/// Temporär arbetskatalog för ett fall.
pub fn unique_dir(tag: &str) -> Result<PathBuf> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let safe: String = tag.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let dir = std::env::temp_dir().join("rustyfem-check").join(format!("{safe}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| FemError(format!("kunde inte skapa {}: {e}", dir.display())))?;
    Ok(dir)
}

/// Resultatet av ett fall.
pub struct CaseOutcome {
    pub report: CaseReport,
    pub nodes: usize,
    /// Baslinjen skrevs om från den här körningen.
    pub updated: bool,
}

/// Kör ett fall. `update`: skriv om `fall.ref` från RustyFEM; annars jämförs
/// mot den sparade baslinjen.
pub fn check_case(inp: &Path, update: bool) -> Result<CaseOutcome> {
    let spec = case::load_spec(inp)?;
    let (ours, m) = run_rustyfem_spec(inp, &spec)?;
    let refp = case::ref_path(inp);
    if update {
        std::fs::write(&refp, ours.to_text()).map_err(|e| FemError(format!("kunde inte skriva {}: {e}", refp.display())))?;
    }
    let text = std::fs::read_to_string(&refp)
        .map_err(|_| FemError(format!("{} saknas – kör 'rustyfem baseline {}' först", refp.display(), inp.display())))?;
    let reference = ResultSet::parse(&text)?;
    let report = compare::compare(&ours, &reference, &spec);
    Ok(CaseOutcome { report, nodes: m.model.nodes.len(), updated: update })
}
