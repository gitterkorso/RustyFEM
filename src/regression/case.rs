//! Regressionsfall: `tests/cases/<område>/<fall>.inp` med sidofilen
//! `<fall>.toml` (tolerans och storheter) och baslinjen `<fall>.ref`
//! (RustyFEMs resultatformat, skriven av `rustyfem baseline`).
//!
//! Sidofilen är en liten TOML-delmängd:
//!
//! ```toml
//! description = "Konsol i plan spänning med tryck"
//! tol = 1e-4                 # relativt fel, normerat mot största |värde| i fältet
//! fields = ["U", "RF", "S"]  # storheter som jämförs (FRAMEVALUE = egenvärde/tid,
//!                            # H:NAMN = sista history-värdet, HA:NAMN = alla)
//! frames = "last"            # "last" (sista framen i varje steg) eller "all"
//! tol.S = 1e-3               # egen tolerans per fält (valfritt)
//! atol.RF = 1e-8             # absolut tolerans i fältets enheter nära noll
//! sign_free = ["U"]          # fält där tecknet får skifta per frame (modformer)
//! floor = 1e-8               # normera minst mot floor · största |värde| i fältet
//!                            # över stegets alla frames (valfritt, för brus i
//!                            # komponenter som är ≈ 0 i en mod)
//! floor_elem = 1e-6          # för elementfält (S, PEEQ, SDEG, ...) mot största |värde|
//!                            # över fallets alla steg, t.ex. för skadevariabler
//!                            # som är ≈ 0 före skadeinitiering
//! steps = [2, 3]             # jämför bara dessa steg (valfritt, 1-baserat)
//! global_inp = "global.inp"  # submodell: globalmodellen körs först (--global)
//! old_inp = "gammal.inp"     # omstart: den första analysen körs först (--oldjob)
//! ```
//!
//! Bara .inp-filer med en sidofil räknas som fall; inkluderade filer ska inte ha någon.

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
    /// Undre gräns för normeringen relativt fältets största värde i steget (0 = av).
    pub floor: f64,
    /// Som `floor` men för elementfält (0 = av).
    pub floor_elem: f64,
    /// Steg som jämförs (1-baserat); tom = alla.
    pub steps: Vec<usize>,
    /// Globalmodellen för en submodell (relativt fallets katalog).
    pub global_inp: Option<String>,
    /// Den ursprungliga analysen för ett omstartsfall (relativt fallets katalog).
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

/// Tolkar sidofilens text.
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
            return fail(format!("{name} rad {}: skriv 'nyckel = värde'", no + 1));
        };
        let (k, v) = (k.trim(), v.trim());
        let num = |v: &str| v.parse::<f64>().map_err(|_| FemError(format!("{name} rad {}: '{v}' är inget tal", no + 1)));
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
            _ => return fail(format!("{name} rad {}: okänd nyckel '{k}'", no + 1)),
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

/// Läser sidofilen (standardvärden om den saknas).
pub fn load_spec(inp: &Path) -> Result<CaseSpec> {
    let p = spec_path(inp);
    match std::fs::read_to_string(&p) {
        Ok(t) => parse_spec(&t, &p.display().to_string()),
        Err(_) => Ok(CaseSpec::default()),
    }
}

/// Alla fall (.inp med sidofil) under `root`, sorterade.
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

/// "område/fall" för utskrift.
pub fn case_name(inp: &Path) -> String {
    let stem = inp.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    match inp.parent().and_then(|p| p.file_name()) {
        Some(a) => format!("{}/{stem}", a.to_string_lossy()),
        None => stem,
    }
}
