//! Delar upp en .inp-fil i nyckelordsblock och följer *INCLUDE.
//!
//! - Rader som börjar med `**` är kommentarer; tomma rader hoppas över.
//! - En nyckelordsrad som slutar med komma fortsätter på nästa rad.
//! - Datarader delas på komma; en datarad som slutar med komma markeras
//!   `continued` (används t.ex. av *ELEMENT med många noder).
//! - Nyckelord och parameternamn görs om till versaler; parametervärden
//!   behålls som de står (utan citattecken).

use crate::fem::model::{DataLine, RawKeyword};
use crate::model::{fail, Result};
use std::path::{Path, PathBuf};

fn normalize_name(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_uppercase()
}

fn split_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in s.chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn parse_keyword_line(text: &str, file: &str, line: usize) -> RawKeyword {
    let parts = split_params(text);
    let name = normalize_name(&parts[0]);
    let mut params = Vec::new();
    for p in &parts[1..] {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        match p.split_once('=') {
            Some((k, v)) => params.push((normalize_name(k), v.trim().to_string())),
            None => params.push((normalize_name(p), String::new())),
        }
    }
    RawKeyword { name, params, data: Vec::new(), file: file.to_string(), line }
}

/// Ger innehållet för en sökväg (filresolver). Standard är filsystemet; appen
/// kan ge en egen (t.ex. filer som importerats till iOS-sandlådan).
pub type Resolver<'a> = &'a dyn Fn(&Path) -> Result<Vec<u8>>;

/// Filsystemet som resolver.
pub fn fs_resolver(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| crate::model::FemError(format!("kunde inte läsa {}: {e}", path.display())))
}

fn read_text(path: &Path, resolve: Resolver) -> Result<String> {
    let bytes = resolve(path)?;
    let s = String::from_utf8_lossy(&bytes).to_string();
    Ok(s.strip_prefix('\u{feff}').map(|t| t.to_string()).unwrap_or(s))
}

/// Läser filen och alla *INCLUDE-filer. `included` får de inkluderade sökvägarna.
pub fn blocks_from_file(path: &Path, included: &mut Vec<PathBuf>, depth: usize) -> Result<Vec<RawKeyword>> {
    blocks_from_file_with(path, &fs_resolver, included, depth)
}

/// Som `blocks_from_file`, med filerna från `resolve`.
pub fn blocks_from_file_with(path: &Path, resolve: Resolver, included: &mut Vec<PathBuf>, depth: usize) -> Result<Vec<RawKeyword>> {
    if depth > 20 {
        return fail(format!("{}: för djup *INCLUDE-nästling", path.display()));
    }
    let text = read_text(path, resolve)?;
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    blocks_from_str_with(&text, &name, &dir, resolve, included, depth)
}

/// Delar upp text i block. `dir` används för relativa *INCLUDE-sökvägar.
pub fn blocks_from_str(text: &str, file: &str, dir: &Path, included: &mut Vec<PathBuf>, depth: usize) -> Result<Vec<RawKeyword>> {
    blocks_from_str_with(text, file, dir, &fs_resolver, included, depth)
}

/// Som `blocks_from_str`, med *INCLUDE-filerna från `resolve`.
pub fn blocks_from_str_with(text: &str, file: &str, dir: &Path, resolve: Resolver, included: &mut Vec<PathBuf>, depth: usize) -> Result<Vec<RawKeyword>> {
    let mut out: Vec<RawKeyword> = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i].trim_end_matches(['\r', '\n']);
        let no = i + 1;
        i += 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with("**") {
            continue;
        }
        if let Some(rest) = t.strip_prefix('*') {
            let mut text = rest.to_string();
            while text.trim_end().ends_with(',') && i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with("**") {
                    i += 1;
                    continue;
                }
                text.push_str(next);
                i += 1;
            }
            let kw = parse_keyword_line(&text, file, no);
            if kw.name == "INCLUDE" {
                let Some(inp) = kw.param("INPUT") else {
                    return fail(format!("{file} rad {no}: *INCLUDE saknar INPUT="));
                };
                let p = dir.join(inp.trim().trim_matches('"'));
                included.push(p.clone());
                out.extend(blocks_from_file_with(&p, resolve, included, depth + 1)?);
                continue;
            }
            out.push(kw);
        } else {
            let Some(kw) = out.last_mut() else {
                return fail(format!("{file} rad {no}: datarad före första nyckelordet"));
            };
            let continued = t.ends_with(',');
            let body = if continued { &t[..t.len() - 1] } else { t };
            let fields = if kw.name == "HEADING" { vec![body.to_string()] } else { body.split(',').map(|f| f.trim().to_string()).collect() };
            kw.data.push(DataLine { fields, line: no, continued });
        }
    }
    Ok(out)
}
