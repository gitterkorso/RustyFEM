//! Parameterized input: `*PARAMETER` and `<name>`.
//!
//! The data lines in *PARAMETER are simple Python statements `name = expression`, one per
//! line. The expressions may use numbers, strings ('...' or "..."), previously
//! defined parameters, `+ − * / // % **`, parentheses and the functions
//! in Python's `math` (with or without `math.`): sin, cos, tan, asin, acos,
//! atan, atan2, sinh, cosh, tanh, sqrt, exp, log (also with base), log10, pow,
//! fabs, floor, ceil, radians, degrees, hypot plus abs, min, max, int, float,
//! round and str, and the constants pi and e. Integers and floats follow
//! Python 3 (1/2 = 0.5, 7//2 = 3, 2**3 = 8).
//!
//! After the definition, `<name>` in parameter values on keyword lines
//! and in data lines is replaced by the value (floats like Python's `repr`, e.g. `0.1`,
//! `210000.0`). The *PARAMETER blocks are removed before the keywords are parsed.
//! `*PARAMETER TABLE`, `*TABLE COLLECTION`, `*PARAMETER DEPENDENCE` and
//! `INPUT=` give an error (not supported yet).

use crate::fem::model::RawKeyword;
use crate::model::{fail, Result};
use std::collections::HashMap;

/// A parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    fn num(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(_) => None,
        }
    }

    /// The value as text (Python's str/repr for numbers).
    pub fn text(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => py_float(*f),
            Value::Str(s) => s.clone(),
        }
    }
}

/// Float like Python's repr: the shortest text that gives the same number, always with
/// a decimal point or exponent.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = f.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        // like Python: 1e-05, 1.5e+20
        let s = format!("{f:e}");
        let (m, e) = s.split_once('e').unwrap();
        let ex: i32 = e.parse().unwrap_or(0);
        return format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs());
    }
    let s = format!("{f}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(Value),
    Str(String),
    Id(String),
    Op(String),
    LPar,
    RPar,
    Comma,
}

fn tokenize(s: &str) -> std::result::Result<Vec<Tok>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() || (ch == '.' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            let mut float = false;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.' || c[i] == '_') {
                float |= c[i] == '.';
                i += 1;
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let j = i + 1 + usize::from(c.get(i + 1).is_some_and(|&x| x == '+' || x == '-'));
                if c.get(j).is_some_and(|d| d.is_ascii_digit()) {
                    float = true;
                    i = j;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let t: String = c[st..i].iter().filter(|&&x| x != '_').collect();
            let v = if float { t.parse::<f64>().map(Value::Float).map_err(|_| format!("'{t}' is not a number"))? } else { t.parse::<i64>().map(Value::Int).map_err(|_| format!("'{t}' is not a number"))? };
            out.push(Tok::Num(v));
        } else if ch.is_alphabetic() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_' || c[i] == '.') {
                i += 1;
            }
            out.push(Tok::Id(c[st..i].iter().collect()));
        } else if ch == '\'' || ch == '"' {
            let q = ch;
            let st = i + 1;
            i += 1;
            while i < c.len() && c[i] != q {
                i += 1;
            }
            if i >= c.len() {
                return Err("the string lacks a closing quote".into());
            }
            out.push(Tok::Str(c[st..i].iter().collect()));
            i += 1;
        } else if ch == '(' {
            out.push(Tok::LPar);
            i += 1;
        } else if ch == ')' {
            out.push(Tok::RPar);
            i += 1;
        } else if ch == ',' {
            out.push(Tok::Comma);
            i += 1;
        } else if ch == '*' && c.get(i + 1) == Some(&'*') {
            out.push(Tok::Op("**".into()));
            i += 2;
        } else if ch == '/' && c.get(i + 1) == Some(&'/') {
            out.push(Tok::Op("//".into()));
            i += 2;
        } else if "+-*/%".contains(ch) {
            out.push(Tok::Op(ch.to_string()));
            i += 1;
        } else {
            return Err(format!("unexpected character '{ch}'"));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    t: Vec<Tok>,
    i: usize,
    vars: &'a HashMap<String, Value>,
}

type R = std::result::Result<Value, String>;

fn arith(op: &str, a: Value, b: Value) -> R {
    use Value::*;
    match (op, &a, &b) {
        ("+", Str(x), Str(y)) => return Ok(Str(format!("{x}{y}"))),
        ("*", Str(x), Int(n)) | ("*", Int(n), Str(x)) => return Ok(Str(x.repeat((*n).max(0) as usize))),
        _ => {}
    }
    if let (Int(x), Int(y)) = (&a, &b) {
        let (x, y) = (*x, *y);
        return match op {
            "+" => Ok(Int(x.wrapping_add(y))),
            "-" => Ok(Int(x.wrapping_sub(y))),
            "*" => Ok(Int(x.wrapping_mul(y))),
            "/" if y == 0 => Err("division by zero".into()),
            "/" => Ok(Float(x as f64 / y as f64)),
            "//" if y == 0 => Err("division by zero".into()),
            "//" => Ok(Int(x.div_euclid(y) - if y < 0 && x.rem_euclid(y) != 0 { 1 } else { 0 })),
            "%" if y == 0 => Err("division by zero".into()),
            "%" => Ok(Int(((x % y) + y) % y)),
            "**" if y >= 0 => Ok(Int(x.wrapping_pow(y as u32))),
            "**" => Ok(Float((x as f64).powf(y as f64))),
            _ => Err(format!("unknown operator {op}")),
        };
    }
    let (Some(x), Some(y)) = (a.num(), b.num()) else {
        return Err(format!("the operator {op} cannot be used on strings"));
    };
    match op {
        "+" => Ok(Float(x + y)),
        "-" => Ok(Float(x - y)),
        "*" => Ok(Float(x * y)),
        "/" if y == 0.0 => Err("division by zero".into()),
        "/" => Ok(Float(x / y)),
        "//" if y == 0.0 => Err("division by zero".into()),
        "//" => Ok(Float((x / y).floor())),
        "%" if y == 0.0 => Err("division by zero".into()),
        "%" => Ok(Float(x - y * (x / y).floor())),
        "**" => Ok(Float(x.powf(y))),
        _ => Err(format!("unknown operator {op}")),
    }
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn op(&self) -> Option<&str> {
        match self.peek() {
            Some(Tok::Op(o)) => Some(o.as_str()),
            _ => None,
        }
    }

    fn expr(&mut self) -> R {
        let mut v = self.term()?;
        while let Some(o) = self.op().filter(|o| *o == "+" || *o == "-").map(|o| o.to_string()) {
            self.i += 1;
            let r = self.term()?;
            v = arith(&o, v, r)?;
        }
        Ok(v)
    }

    fn term(&mut self) -> R {
        let mut v = self.unary()?;
        while let Some(o) = self.op().filter(|o| matches!(*o, "*" | "/" | "//" | "%")).map(|o| o.to_string()) {
            self.i += 1;
            let r = self.unary()?;
            v = arith(&o, v, r)?;
        }
        Ok(v)
    }

    fn unary(&mut self) -> R {
        match self.op() {
            Some("-") => {
                self.i += 1;
                let v = self.unary()?;
                match v {
                    Value::Int(i) => Ok(Value::Int(-i)),
                    Value::Float(f) => Ok(Value::Float(-f)),
                    Value::Str(_) => Err("unary minus on a string".into()),
                }
            }
            Some("+") => {
                self.i += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> R {
        let b = self.atom()?;
        if self.op() == Some("**") {
            self.i += 1;
            let e = self.unary()?; // right-associative, binds tighter than a unary minus on the left
            return arith("**", b, e);
        }
        Ok(b)
    }

    fn atom(&mut self) -> R {
        let t = self.peek().cloned().ok_or("the expression ends prematurely")?;
        self.i += 1;
        match t {
            Tok::Num(v) => Ok(v),
            Tok::Str(s) => Ok(Value::Str(s)),
            Tok::LPar => {
                let v = self.expr()?;
                if self.peek() != Some(&Tok::RPar) {
                    return Err("')' is missing".into());
                }
                self.i += 1;
                Ok(v)
            }
            Tok::Id(name) => {
                if self.peek() == Some(&Tok::LPar) {
                    self.i += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RPar) {
                        loop {
                            args.push(self.expr()?);
                            match self.peek() {
                                Some(Tok::Comma) => self.i += 1,
                                Some(Tok::RPar) => break,
                                _ => return Err("',' or ')' is missing in the function call".into()),
                            }
                        }
                    }
                    self.i += 1;
                    return call(&name, &args);
                }
                let n = name.strip_prefix("math.").unwrap_or(&name);
                if let Some(v) = self.vars.get(&name) {
                    return Ok(v.clone());
                }
                match n {
                    "pi" => Ok(Value::Float(std::f64::consts::PI)),
                    "e" => Ok(Value::Float(std::f64::consts::E)),
                    "True" => Ok(Value::Int(1)),
                    "False" => Ok(Value::Int(0)),
                    _ => Err(format!("parameter '{name}' is not defined")),
                }
            }
            t => Err(format!("unexpected token {t:?}")),
        }
    }
}

fn call(name: &str, a: &[Value]) -> R {
    let n = name.strip_prefix("math.").unwrap_or(name);
    let x = |k: usize| -> std::result::Result<f64, String> { a.get(k).and_then(|v| v.num()).ok_or(format!("{name}() requires a number as argument {}", k + 1)) };
    let f = |v: f64| Ok(Value::Float(v));
    match (n, a.len()) {
        ("sin", 1) => f(x(0)?.sin()),
        ("cos", 1) => f(x(0)?.cos()),
        ("tan", 1) => f(x(0)?.tan()),
        ("asin", 1) => f(x(0)?.asin()),
        ("acos", 1) => f(x(0)?.acos()),
        ("atan", 1) => f(x(0)?.atan()),
        ("atan2", 2) => f(x(0)?.atan2(x(1)?)),
        ("sinh", 1) => f(x(0)?.sinh()),
        ("cosh", 1) => f(x(0)?.cosh()),
        ("tanh", 1) => f(x(0)?.tanh()),
        ("sqrt", 1) if x(0)? < 0.0 => Err("sqrt of a negative number".into()),
        ("sqrt", 1) => f(x(0)?.sqrt()),
        ("exp", 1) => f(x(0)?.exp()),
        ("log", 1) => f(x(0)?.ln()),
        ("log", 2) => f(x(0)?.ln() / x(1)?.ln()),
        ("log10", 1) => f(x(0)?.log10()),
        ("pow", 2) => f(x(0)?.powf(x(1)?)),
        ("fabs", 1) => f(x(0)?.abs()),
        ("hypot", 2) => f(x(0)?.hypot(x(1)?)),
        ("radians", 1) => f(x(0)?.to_radians()),
        ("degrees", 1) => f(x(0)?.to_degrees()),
        ("floor", 1) => Ok(Value::Int(x(0)?.floor() as i64)),
        ("ceil", 1) => Ok(Value::Int(x(0)?.ceil() as i64)),
        ("abs", 1) => match &a[0] {
            Value::Int(i) => Ok(Value::Int(i.abs())),
            _ => f(x(0)?.abs()),
        },
        ("int", 1) => match &a[0] {
            Value::Str(s) => s.trim().parse::<i64>().map(Value::Int).map_err(|_| format!("int('{s}') is not an integer")),
            _ => Ok(Value::Int(x(0)?.trunc() as i64)),
        },
        ("float", 1) => match &a[0] {
            Value::Str(s) => s.trim().parse::<f64>().map(Value::Float).map_err(|_| format!("float('{s}') is not a number")),
            _ => f(x(0)?),
        },
        ("round", 1) => Ok(Value::Int(round_half_even(x(0)?) as i64)),
        ("round", 2) => {
            let s = 10f64.powi(x(1)? as i32);
            f(round_half_even(x(0)? * s) / s)
        }
        ("str", 1) => Ok(Value::Str(a[0].text())),
        ("min", k) | ("max", k) if k >= 1 => {
            let mut best = a[0].clone();
            for v in &a[1..] {
                let (p, q) = (best.num().ok_or("min/max requires numbers")?, v.num().ok_or("min/max requires numbers")?);
                if (n == "min" && q < p) || (n == "max" && q > p) {
                    best = v.clone();
                }
            }
            Ok(best)
        }
        _ => Err(format!("function {name}() with {} arguments is not supported", a.len())),
    }
}

/// Rounding like Python 3 (to even at exactly half).
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Evaluates an expression with the parameters `vars`.
pub fn eval(expr: &str, vars: &HashMap<String, Value>) -> std::result::Result<Value, String> {
    let t = tokenize(expr)?;
    if t.is_empty() {
        return Err("empty expression".into());
    }
    let mut p = Parser { t, i: 0, vars };
    let v = p.expr()?;
    if p.i != p.t.len() {
        return Err(format!("unexpected continuation after the expression: {:?}", p.t[p.i]));
    }
    Ok(v)
}

/// Replaces `<name>` in `s`.
fn substitute(s: &str, vars: &HashMap<String, Value>, kw: &RawKeyword, line: usize) -> Result<String> {
    if !s.contains('<') {
        return Ok(s.to_string());
    }
    let mut out = String::new();
    let mut rest = s;
    while let Some(a) = rest.find('<') {
        let Some(b) = rest[a..].find('>') else { break };
        let name = &rest[a + 1..a + b];
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.push_str(&rest[..a + 1]);
            rest = &rest[a + 1..];
            continue;
        }
        let Some(v) = vars.get(name) else {
            return kw.err_at(line, format!("parameter <{name}> is not defined (*PARAMETER)"));
        };
        out.push_str(&rest[..a]);
        out.push_str(&v.text());
        rest = &rest[a + b + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Parses *PARAMETER blocks and replaces `<name>` in all following blocks.
/// Files without *PARAMETER are left untouched.
pub fn expand(blocks: Vec<RawKeyword>) -> Result<Vec<RawKeyword>> {
    if !blocks.iter().any(|k| k.name.starts_with("PARAMETER") || k.name == "TABLE COLLECTION") {
        return Ok(blocks);
    }
    let mut vars: HashMap<String, Value> = HashMap::new();
    let mut out = Vec::with_capacity(blocks.len());
    for mut kw in blocks {
        match kw.name.as_str() {
            "PARAMETER" => {
                if kw.has("INPUT") {
                    return kw.err("*PARAMETER, INPUT= is not supported yet (write the statements in the .inp file)");
                }
                for d in &kw.data {
                    let text = d.fields.join(",");
                    let t = text.trim();
                    if t.is_empty() || t.starts_with('#') {
                        continue;
                    }
                    let Some((name, expr)) = t.split_once('=') else {
                        return kw.err_at(d.line, format!("'{t}' must be 'name = expression'"));
                    };
                    let name = name.trim();
                    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') || name.starts_with(|c: char| c.is_ascii_digit()) {
                        return kw.err_at(d.line, format!("'{name}' is not a valid parameter name"));
                    }
                    let expr = expr.split('#').next().unwrap_or("");
                    let v = eval(expr, &vars).map_err(|e| crate::model::FemError(format!("{} line {} (*PARAMETER): {e}", kw.file, d.line)))?;
                    vars.insert(name.to_string(), v);
                }
            }
            "PARAMETER TABLE" | "PARAMETER TABLE TYPE" | "TABLE COLLECTION" | "PARAMETER DEPENDENCE" | "PARAMETER SHAPE VARIATION" => {
                return kw.err("the keyword is not supported yet (only *PARAMETER)");
            }
            _ => {
                if vars.is_empty() {
                    out.push(kw);
                    continue;
                }
                let line = kw.line;
                let mut params = std::mem::take(&mut kw.params);
                for (_, v) in params.iter_mut() {
                    *v = substitute(v, &vars, &kw, line)?;
                }
                kw.params = params;
                let mut data = std::mem::take(&mut kw.data);
                for d in data.iter_mut() {
                    for f in d.fields.iter_mut() {
                        *f = substitute(f, &vars, &kw, d.line)?.trim().to_string();
                    }
                }
                kw.data = data;
                out.push(kw);
            }
        }
    }
    if out.is_empty() {
        return fail("the file contains only *PARAMETER");
    }
    Ok(out)
}
