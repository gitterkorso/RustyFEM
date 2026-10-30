//! The `.rusty` text format: one command per line, `#` starts a comment.
//!
//! ```text
//! material steel E=210e9 nu=0.3
//! plate x=0..2 y=0..0.2 t=0.01 size=0.02
//! fix x=0
//! load x=2 fy=-1000
//! ```
//! See docs/REFERENCE.md for all commands.

use crate::model::{fail, Dim, Material, Model, Result, Sel, RX, RY, RZ, UX, UY, UZ};
use crate::section::BeamSection;

struct Args {
    line: usize,
    items: Vec<(Option<String>, String)>,
    used: Vec<bool>,
}

fn err<T>(line: usize, msg: impl AsRef<str>) -> Result<T> {
    fail(format!("line {line}: {}", msg.as_ref()))
}

fn parse_num(line: usize, s: &str) -> Result<f64> {
    s.parse::<f64>().or_else(|_| err(line, format!("'{s}' is not a number")))
}

fn parse_range(line: usize, s: &str) -> Result<(f64, f64)> {
    match s.split_once("..") {
        Some((a, b)) => {
            let (a, b) = (parse_num(line, a)?, parse_num(line, b)?);
            Ok((a.min(b), a.max(b)))
        }
        None => {
            let v = parse_num(line, s)?;
            Ok((v, v))
        }
    }
}

fn parse_list(line: usize, s: &str) -> Result<Vec<f64>> {
    let inner = s.strip_prefix('(').and_then(|t| t.strip_suffix(')')).ok_or_else(|| {
        crate::model::FemError(format!("line {line}: '{s}' must be a list in parentheses, e.g. (1,2)"))
    })?;
    inner.split(',').map(|t| parse_num(line, t)).collect()
}

fn parse_point(line: usize, s: &str) -> Result<([f64; 3], usize)> {
    let v = parse_list(line, s)?;
    match v.len() {
        2 => Ok(([v[0], v[1], 0.0], 2)),
        3 => Ok(([v[0], v[1], v[2]], 3)),
        _ => err(line, format!("the point '{s}' must have 2 or 3 coordinates")),
    }
}

impl Args {
    fn get(&mut self, key: &str) -> Option<String> {
        let i = self.items.iter().position(|(k, _)| k.as_deref() == Some(key))?;
        self.used[i] = true;
        Some(self.items[i].1.clone())
    }

    fn num(&mut self, key: &str) -> Result<Option<f64>> {
        match self.get(key) {
            Some(v) => Ok(Some(parse_num(self.line, &v)?)),
            None => Ok(None),
        }
    }

    fn req(&mut self, key: &str, what: &str) -> Result<f64> {
        match self.num(key)? {
            Some(v) => Ok(v),
            None => err(self.line, format!("{key}= is missing ({what})")),
        }
    }

    fn range(&mut self, key: &str) -> Result<(f64, f64)> {
        match self.get(key) {
            Some(v) => parse_range(self.line, &v),
            None => err(self.line, format!("{key}= is missing, e.g. {key}=0..1")),
        }
    }

    fn flag(&mut self, word: &str) -> bool {
        match self.items.iter().position(|(k, v)| k.is_none() && v == word) {
            Some(i) => {
                self.used[i] = true;
                true
            }
            None => false,
        }
    }

    fn points(&mut self) -> Result<Vec<[f64; 3]>> {
        let mut pts = Vec::new();
        for i in 0..self.items.len() {
            if self.items[i].0.is_none() && self.items[i].1.starts_with('(') {
                self.used[i] = true;
                pts.push(parse_point(self.line, &self.items[i].1)?.0);
            }
        }
        Ok(pts)
    }

    fn selection(&mut self) -> Result<Sel> {
        if self.flag("all") {
            return Ok(Sel::All);
        }
        if let Some(p) = self.get("at") {
            return Ok(Sel::Point(parse_point(self.line, &p)?.0));
        }
        let mut conds = Vec::new();
        for (axis, key) in ["x", "y", "z"].iter().enumerate() {
            if let Some(v) = self.get(key) {
                let (lo, hi) = parse_range(self.line, &v)?;
                conds.push((axis, lo, hi));
            }
        }
        if conds.is_empty() {
            return err(self.line, "selection is missing: specify e.g. x=0, y=0..1, at=(1,2) or all");
        }
        Ok(Sel::Box(conds))
    }

    fn finish(&self) -> Result<()> {
        for (i, (k, v)) in self.items.iter().enumerate() {
            if !self.used[i] {
                return match k {
                    Some(k) => err(self.line, format!("unknown parameter '{k}='")),
                    None => err(self.line, format!("unexpected word '{v}'")),
                };
            }
        }
        Ok(())
    }
}

/// Removes the comment and whitespace inside parentheses.
fn normalize(raw: &str) -> String {
    let s = raw.split('#').next().unwrap_or("");
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if depth > 0 && c.is_whitespace() {
            continue;
        }
        out.push(c);
    }
    out
}

struct Line {
    no: usize,
    cmd: String,
    args: Args,
}

fn tokenize(text: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let norm = normalize(raw);
        let mut toks = norm.split_whitespace();
        let Some(cmd) = toks.next() else { continue };
        let items: Vec<(Option<String>, String)> = toks
            .map(|t| match t.split_once('=') {
                Some((k, v)) => (Some(k.to_lowercase()), v.to_string()),
                None => (None, t.to_lowercase()),
            })
            .collect();
        let used = vec![false; items.len()];
        lines.push(Line { no: i + 1, cmd: cmd.to_lowercase(), args: Args { line: i + 1, items, used } });
    }
    lines
}

fn infer_dim(lines: &[Line]) -> Result<Dim> {
    for l in lines {
        if l.cmd == "model" {
            return match l.args.items.first().map(|i| i.1.as_str()) {
                Some("2d") => Ok(Dim::Two),
                Some("3d") => Ok(Dim::Three),
                _ => err(l.no, "write 'model 2d' or 'model 3d'"),
            };
        }
    }
    for l in lines {
        if l.cmd == "box" {
            return Ok(Dim::Three);
        }
        for (k, v) in &l.args.items {
            if v.starts_with('(') && (k.is_none() || k.as_deref() == Some("at")) && parse_point(l.no, v).map(|p| p.1 == 3).unwrap_or(false) {
                return Ok(Dim::Three);
            }
        }
    }
    Ok(Dim::Two)
}

fn parse_section(line: usize, s: &str) -> Result<BeamSection> {
    let (name, rest) = s.split_once('(').ok_or_else(|| {
        crate::model::FemError(format!("line {line}: a cross-section is written e.g. rect(0.05,0.1)"))
    })?;
    let v = parse_list(line, &format!("({rest}"))?;
    let need = |n: usize| if v.len() == n { Ok(()) } else { err(line, format!("{name}(...) must have {n} values")) };
    match name {
        "rect" => need(2).map(|_| BeamSection::rect(v[0], v[1])),
        "circle" => need(1).map(|_| BeamSection::circle(v[0])),
        "pipe" => need(2).map(|_| BeamSection::pipe(v[0], v[1])),
        "ibeam" => need(4).map(|_| BeamSection::ibeam(v[0], v[1], v[2], v[3])),
        "general" => need(4).map(|_| BeamSection::general(v[0], v[1], v[2], v[3])),
        _ => err(line, format!("unknown cross-section '{name}' (rect, circle, pipe, ibeam, general)")),
    }
}

fn pick_material(a: &mut Args, m: &mut Model, current: &mut Option<usize>, line: usize) -> Result<usize> {
    if let Some(name) = a.get("material") {
        let name = name.to_lowercase();
        return match m.materials.iter().position(|x| x.name.to_lowercase() == name) {
            Some(i) => Ok(i),
            None => err(line, format!("material '{name}' is not defined")),
        };
    }
    Ok(*current.get_or_insert_with(|| m.material(Material::steel())))
}

/// Parses a model in text format.
pub fn parse(text: &str) -> Result<Model> {
    let mut lines = tokenize(text);
    let dim = infer_dim(&lines)?;
    let mut m = if dim == Dim::Two { Model::new_2d() } else { Model::new_3d() };
    let mut current: Option<usize> = None;

    for l in lines.iter_mut() {
        let a = &mut l.args;
        let line = l.no;
        match l.cmd.as_str() {
            "model" => {
                a.used.iter_mut().for_each(|u| *u = true);
            }
            "plane" => {
                if a.flag("strain") {
                    m.plane_strain = true;
                } else if a.flag("stress") {
                    m.plane_strain = false;
                } else {
                    return err(line, "write 'plane stress' or 'plane strain'");
                }
            }
            "elements" => {
                if a.flag("tri") {
                    m.triangles = true;
                } else if a.flag("quad") {
                    m.triangles = false;
                } else {
                    return err(line, "write 'elements quad' or 'elements tri'");
                }
            }
            "material" => {
                let Some((None, name)) = a.items.first().cloned() else {
                    return err(line, "the material needs a name, e.g. 'material steel'");
                };
                a.used[0] = true;
                let base = match name.as_str() {
                    "steel" | "stål" => Some(Material::steel()),
                    "aluminium" | "aluminum" => Some(Material::aluminium()),
                    _ => None,
                };
                let e = a.num("e")?;
                let nu = a.num("nu")?;
                let rho = a.num("rho")?;
                let mut mat = match (base, e) {
                    (Some(b), _) => b,
                    (None, Some(_)) => Material::new(&name, 0.0, 0.3),
                    (None, None) => return err(line, format!("material '{name}' needs E=")),
                };
                mat.name = name;
                if let Some(e) = e {
                    mat.e = e;
                }
                if let Some(nu) = nu {
                    mat.nu = nu;
                }
                if let Some(rho) = rho {
                    mat.density = rho;
                }
                current = Some(m.material(mat));
            }
            "plate" => {
                let (x0, x1) = a.range("x")?;
                let (y0, y1) = a.range("y")?;
                let t = a.req("t", "thickness")?;
                let size = a.req("size", "element size")?;
                let mat = pick_material(a, &mut m, &mut current, line)?;
                m.plate([x0, y0], [x1, y1], t, size, mat).map_err(|e| crate::model::FemError(format!("line {line}: {e}")))?;
            }
            "quad" => {
                let p = a.points()?;
                if p.len() != 4 {
                    return err(line, "quad needs four corners, e.g. quad (0,0) (2,0) (2,1) (0,1)");
                }
                let t = a.req("t", "thickness")?;
                let size = a.req("size", "element size")?;
                let mat = pick_material(a, &mut m, &mut current, line)?;
                let c = [[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]];
                m.quad_region(c, t, size, mat).map_err(|e| crate::model::FemError(format!("line {line}: {e}")))?;
            }
            "box" => {
                let (x0, x1) = a.range("x")?;
                let (y0, y1) = a.range("y")?;
                let (z0, z1) = a.range("z")?;
                let size = a.req("size", "element size")?;
                let mat = pick_material(a, &mut m, &mut current, line)?;
                m.box_solid([x0, y0, z0], [x1, y1, z1], size, mat).map_err(|e| crate::model::FemError(format!("line {line}: {e}")))?;
            }
            "beam" => {
                let p = a.points()?;
                if p.len() < 2 {
                    return err(line, "beam needs at least two points, e.g. beam (0,0) (2,0) section=rect(0.05,0.1)");
                }
                let sec = match a.get("section") {
                    Some(s) => parse_section(line, &s)?,
                    None => return err(line, "section= is missing, e.g. section=rect(0.05,0.1)"),
                };
                let n = a.num("n")?.unwrap_or(10.0).max(1.0) as usize;
                let up = match a.get("up") {
                    Some(s) => Some(parse_point(line, &s)?.0),
                    None => None,
                };
                let mat = pick_material(a, &mut m, &mut current, line)?;
                for w in p.windows(2) {
                    m.beam(w[0], w[1], sec.clone(), mat, n, up).map_err(|e| crate::model::FemError(format!("line {line}: {e}")))?;
                }
            }
            "truss" => {
                let p = a.points()?;
                if p.len() < 2 {
                    return err(line, "truss needs at least two points");
                }
                let area = a.req("area", "cross-section area")?;
                let mat = pick_material(a, &mut m, &mut current, line)?;
                for w in p.windows(2) {
                    m.truss(w[0], w[1], area, mat).map_err(|e| crate::model::FemError(format!("line {line}: {e}")))?;
                }
            }
            "fix" => {
                let s = a.selection()?;
                m.fix(s);
            }
            "pin" => {
                let s = a.selection()?;
                m.pin(s);
            }
            "support" => {
                let s = a.selection()?;
                let mut dofs = Vec::new();
                for (d, name) in [(UX, "ux"), (UY, "uy"), (UZ, "uz"), (RX, "rx"), (RY, "ry"), (RZ, "rz")] {
                    if a.flag(name) {
                        dofs.push((d, 0.0));
                    } else if let Some(v) = a.num(name)? {
                        dofs.push((d, v));
                    }
                }
                if dofs.is_empty() {
                    return err(line, "specify which degrees of freedom are constrained, e.g. 'support x=0 ux' or 'uy=0.001'");
                }
                m.support(s, &dofs);
            }
            "load" => {
                let s = a.selection()?;
                let mut f = [0.0; 6];
                for (d, name) in ["fx", "fy", "fz", "mx", "my", "mz"].iter().enumerate() {
                    f[d] = a.num(name)?.unwrap_or(0.0);
                }
                if f.iter().all(|&v| v == 0.0) {
                    return err(line, "specify at least one load component, e.g. fy=-1000");
                }
                m.loads.push(crate::model::Load::Force { sel: s, f });
            }
            "pressure" => {
                let s = a.selection()?;
                let p = a.req("p", "pressure")?;
                m.pressure(s, p);
            }
            "gravity" => {
                let pts = a.points()?;
                let g = if let Some(p) = pts.first() {
                    *p
                } else {
                    let mag = a.num("g")?.unwrap_or(9.81);
                    let gx = a.num("gx")?;
                    let gy = a.num("gy")?;
                    let gz = a.num("gz")?;
                    if gx.is_some() || gy.is_some() || gz.is_some() {
                        [gx.unwrap_or(0.0), gy.unwrap_or(0.0), gz.unwrap_or(0.0)]
                    } else if dim == Dim::Two {
                        [0.0, -mag, 0.0]
                    } else {
                        [0.0, 0.0, -mag]
                    }
                };
                m.set_gravity(g);
            }
            other => {
                return err(
                    line,
                    format!("unknown command '{other}' (model, plane, elements, material, plate, quad, box, beam, truss, fix, pin, support, load, pressure, gravity)"),
                )
            }
        }
        a.finish()?;
    }
    Ok(m)
}

pub fn load(path: &std::path::Path) -> Result<Model> {
    let text = std::fs::read_to_string(path).map_err(|e| crate::model::FemError(format!("could not read {}: {e}", path.display())))?;
    parse(&text)
}
