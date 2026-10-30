//! Regression case: `tests/cases/<area>/<case>.inp` with the side file
//! `<case>.toml` (tolerance and quantities) and the baseline `<case>.ref`
//! (RustyFEM's result format, written by `rustyfem baseline`).
//!
//! The side file is a small TOML subset:
//!
//! ```toml
//! description = "Cantilever in plane stress with pressure"
//! tol = 1e-4                 # relative error, normalized by the largest |value| in the field
//! fields = ["U", "RF", "S"]  # quantities compared (FRAMEVALUE = eigenvalue/time,
//!                            # H:NAME = last history value, HA:NAME = all)
//! frames = "last"            # "last" (last frame of each step) or "all"
//! tol.S = 1e-3               # per-field tolerance (optional)
//! atol.RF = 1e-8             # absolute tolerance in the field's units near zero
//! sign_free = ["U"]          # fields whose sign may flip per frame (mode shapes)
//! floor = 1e-8               # normalize by at least floor · largest |value| in the field
//!                            # over all frames of the step (optional, for noise in
//!                            # components that are ≈ 0 in a mode)
//! floor_elem = 1e-6          # for element fields (S, PEEQ, SDEG, ...) against the largest |value|
//!                            # over all steps of the case, e.g. for damage variables
//!                            # that are ≈ 0 before damage initiation
//! steps = [2, 3]             # compare only these steps (optional, 1-based)
//! global_inp = "global.inp"  # submodel: the global model runs first (--global)
//! old_inp = "old.inp"        # restart: the first analysis runs first (--oldjob)
//! ```
//!
//! Only .inp files with a side file count as cases; included files must not have one.

use crate::model::{fail, FemError, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct CaseSpec {
    pub description: String,
    pub tol: f64,
    pub tol_field: BTreeMap<String, f64>,
    /// Absolute tolerances for nodal/element fields (default zero).
    pub atol_field: BTreeMap<String, f64>,
    pub fields: Vec<String>,
    pub all_frames: bool,
    pub sign_free: Vec<String>,
    /// Lower bound for the normalization relative to the field's largest value in the step (0 = off).
    pub floor: f64,
    /// Like `floor` but for element fields (0 = off).
    pub floor_elem: f64,
    /// Steps that are compared (1-based); empty = all.
    pub steps: Vec<usize>,
    /// The global model for a submodel (relative to the case directory).
    pub global_inp: Option<String>,
    /// The original analysis for a restart case (relative to the case directory).
    pub old_inp: Option<String>,
}

impl Default for CaseSpec {
    fn default() -> Self {
        CaseSpec {
            description: String::new(),
            tol: 1e-4,
            tol_field: BTreeMap::new(),
            atol_field: BTreeMap::new(),
            fields: ["U", "UR", "RF", "RM", "S"].iter().map(|s| s.to_string()).collect(),
            all_frames: false,
            sign_free: Vec::new(),
            floor: 0.0,
            floor_elem: 0.0,
            steps: Vec::new(),
            global_inp: None,
            old_inp: None,
        }
    }
}

impl CaseSpec {
    pub fn tol_for(&self, field: &str) -> f64 {
        self.tol_field.get(field).copied().unwrap_or(self.tol)
    }

    /// Minimum reference scale: |difference| <= max(rtol * scale, atol).
    pub fn absolute_norm_floor(&self, field: &str) -> f64 {
        self.atol_field.get(field).filter(|&&v| v > 0.0).map_or(0.0, |v| v / self.tol_for(field))
    }
}

fn parse_list(v: &str) -> Vec<String> {
    let t = v.trim().trim_start_matches('[').trim_end_matches(']');
    t.split(',').map(|s| s.trim().trim_matches('"').trim().to_string()).filter(|s| !s.is_empty()).collect()
}

/// Parses the side file's text.
pub fn parse_spec(text: &str, name: &str) -> Result<CaseSpec> {
    let mut s = CaseSpec::default();
    for (no, raw) in text.lines().enumerate() {
        let line = match raw.find('#') {
            Some(i) if !raw[..i].contains('"') || raw[..i].matches('"').count() % 2 == 0 => &raw[..i],
            _ => raw,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return fail(format!("{name} line {}: write 'key = value'", no + 1));
        };
        let (k, v) = (k.trim(), v.trim());
        let num = |v: &str| v.parse::<f64>().map_err(|_| FemError(format!("{name} line {}: '{v}' is not a number", no + 1)));
        match k {
            "description" => s.description = v.trim_matches('"').to_string(),
            "tol" => s.tol = num(v)?,
            "fields" => s.fields = parse_list(v).into_iter().map(|f| f.to_uppercase()).collect(),
            "frames" => s.all_frames = v.trim_matches('"').eq_ignore_ascii_case("all"),
            "floor" => s.floor = num(v)?,
            "floor_elem" => s.floor_elem = num(v)?,
            "steps" => s.steps = parse_list(v).iter().map(|x| num(x).map(|f| f as usize)).collect::<Result<_>>()?,
            "global_inp" => s.global_inp = Some(v.trim_matches('"').to_string()),
            "old_inp" => s.old_inp = Some(v.trim_matches('"').to_string()),
            "sign_free" => s.sign_free = parse_list(v).into_iter().map(|f| f.to_uppercase()).collect(),
            _ if k.starts_with("tol.") => {
                s.tol_field.insert(k[4..].trim().to_uppercase(), num(v)?);
            }
            _ if k.starts_with("atol.") => {
                let field = k[5..].trim().to_uppercase();
                let value = num(v)?;
                if !value.is_finite() || value < 0.0 || field.is_empty() || field.contains(':') || matches!(field.as_str(), "FRAMEVALUE" | "MODEVALUE") {
                    return fail(format!("{name}: atol requires a nodal/element field and a finite nonnegative value"));
                }
                s.atol_field.insert(field, value);
            }
            _ => return fail(format!("{name} line {}: unknown key '{k}'", no + 1)),
        }
    }
    for field in s.atol_field.keys() {
        if !s.tol_for(field).is_finite() || s.tol_for(field) <= 0.0 {
            return fail(format!("{name}: atol.{field} requires a finite positive relative tolerance"));
        }
    }
    Ok(s)
}

pub fn spec_path(inp: &Path) -> PathBuf {
    inp.with_extension("toml")
}

pub fn ref_path(inp: &Path) -> PathBuf {
    inp.with_extension("ref")
}

/// Reads the side file (default values if it is missing).
pub fn load_spec(inp: &Path) -> Result<CaseSpec> {
    let p = spec_path(inp);
    match std::fs::read_to_string(&p) {
        Ok(t) => parse_spec(&t, &p.display().to_string()),
        Err(_) => Ok(CaseSpec::default()),
    }
}

/// All cases (.inp with side file) under `root`, sorted.
pub fn discover(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("inp")) && spec_path(&p).exists() {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// "area/case" for printing.
pub fn case_name(inp: &Path) -> String {
    let stem = inp.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    match inp.parent().and_then(|p| p.file_name()) {
        Some(a) => format!("{}/{stem}", a.to_string_lossy()),
        None => stem,
    }
}
