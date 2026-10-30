use rustyfem::fem::output::ResultSet;
use rustyfem::results::mises;
use rustyfem::steps::{self, RunOptions};
use rustyfem::regression::{self, case};
use rustyfem::{inp, script};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const HELP: &str = "RustyFEM – FEM solver for .rusty scripts and imported input files in .inp format

Usage:
  rustyfem <model.rusty|model.inp>          run the analysis
  rustyfem run <model.rusty> [--vtk F] [--csv F]
  rustyfem run <model.inp> [--out F] [--all-frames] [-v]
                                           --out writes all results in text format
                    [--oldjob job] [--global global.res]
                                           --oldjob: restart file for *RESTART, READ (job.rfres)
                                           --global: results of the global model for *SUBMODEL
                                           (--out file or .rfres)
  rustyfem check <case.inp ...>             run regression cases and compare with case.ref
  rustyfem check --all [dir]                all cases under tests/cases/ in the test folder (or dir)
  rustyfem baseline <case.inp ...|--all [dir]>
                                           rewrite case.ref from RustyFEM
  rustyfem list                             registered elements, keywords, steps and solvers

Environment variables: RUSTYFEM_SOLVER (linear solver),
RUSTYFEM_USER (shared library with umat_/uel_ for *USER MATERIAL/*USER ELEMENT).

RustyFEM is not intended for real-world structural design; results come with no warranty (see DISCLAIMER.md).";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn is_inp(f: &str) -> bool {
    Path::new(f).extension().is_some_and(|x| x.eq_ignore_ascii_case("inp"))
}

fn run(args: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(first) = args.first() else {
        println!("{HELP}");
        return Ok(true);
    };
    match first.as_str() {
        "-h" | "--help" | "help" => {
            println!("{HELP}");
            Ok(true)
        }
        "run" => {
            let f = args.get(1).ok_or("specify a model file")?;
            if is_inp(f) { cmd_run_inp(f, &args[2..]) } else { cmd_run(f, &args[2..]) }
        }
        "check" => cmd_check(&args[1..], false),
        "baseline" => cmd_check(&args[1..], true),
        "list" => {
            cmd_list();
            Ok(true)
        }
        file if is_inp(file) => cmd_run_inp(file, &args[1..]),
        file => cmd_run(file, &args[1..]),
    }
}

fn cmd_list() {
    println!("Elements:   {}", rustyfem::elements::names().join(", "));
    println!("Keywords:   {}", inp::names().iter().map(|k| format!("*{k}")).collect::<Vec<_>>().join(", "));
    println!("Step types: {}", steps::all().iter().map(|s| s.name()).collect::<Vec<_>>().join(", "));
    println!("Materials:  {}", rustyfem::materials::factories().iter().map(|f| f.name).collect::<Vec<_>>().join(", "));
    println!("Solvers:    {}", rustyfem::solver::all().iter().map(|s| s.name()).collect::<Vec<_>>().join(", "));
}

fn cmd_run(file: &str, rest: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let path = PathBuf::from(file);
    let model = script::load(&path)?;
    let t0 = std::time::Instant::now();
    let res = model.solve()?;
    println!("{}", path.display());
    print!("{}", res.summary(&model));
    println!("Solution time:        {:.3} s", t0.elapsed().as_secs_f64());
    let vtk = opt(rest, "--vtk").map(PathBuf::from).unwrap_or_else(|| path.with_extension("vtk"));
    res.write_vtk(&model, &vtk)?;
    println!("Results for ParaView: {}", vtk.display());
    if let Some(csv) = opt(rest, "--csv") {
        res.write_csv(&model, Path::new(&csv))?;
        println!("Node results:         {csv}");
    }
    Ok(true)
}

/// Short summary of the last frame in each step.
fn summarize(rs: &ResultSet) {
    for st in &rs.steps {
        println!("Step {} ({}, {}): {} frames", st.index, st.name, st.procedure, st.frames.len());
        let Some(fr) = st.frames.last() else { continue };
        if let Some(u) = fr.node.get("U") {
            let best = u.items.iter().map(|(k, c)| (k, c.iter().map(|p| p.1 * p.1).sum::<f64>().sqrt())).fold(None, |a: Option<(&(String, i64), f64)>, b| match a {
                Some(a) if a.1 >= b.1 => Some(a),
                _ => Some(b),
            });
            if let Some(((inst, n), v)) = best {
                println!("  largest displacement |U| = {v:.6e} at node {inst}.{n}");
            }
        }
        if let Some(rf) = fr.node.get("RF") {
            let mut sum = vec![0.0; rf.labels.len()];
            for c in rf.items.values() {
                for &(id, v) in c {
                    sum[id as usize] += v;
                }
            }
            let parts: Vec<String> = rf.labels.iter().zip(&sum).map(|(l, s)| format!("{l}={s:.6e}")).collect();
            println!("  total reactions: {}", parts.join("  "));
        }
        if let Some(s) = fr.elem.get("S") {
            let id = |l: &str| s.labels.iter().position(|x| x == l);
            let ids: Vec<Option<usize>> = ["S11", "S22", "S33", "S12", "S13", "S23"].iter().map(|l| id(l)).collect();
            let mut best = (0.0, String::new());
            for (k, c) in &s.items {
                let mut v = [0.0; 6];
                for (i, li) in ids.iter().enumerate() {
                    if let Some(li) = li {
                        if let Some(p) = c.iter().find(|p| p.0 as usize == *li) {
                            v[i] = p.1;
                        }
                    }
                }
                let m = mises(&v);
                if m > best.0 {
                    best = (m, format!("{}.{} ip {}", k.0, k.1, k.2));
                }
            }
            println!("  highest von Mises = {:.6e} in element {}", best.0, best.1);
        }
    }
}

fn cmd_run_inp(file: &str, rest: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let path = PathBuf::from(file);
    let t0 = std::time::Instant::now();
    // job context: restart file, matrices and substructures are written next to the input file
    let mut job = steps::restart::Job::for_file(&path);
    job.oldjob = opt(rest, "--oldjob").map(PathBuf::from);
    job.global = opt(rest, "--global").map(PathBuf::from);
    let m = steps::restart::read_job(&path, &mut job)?;
    let _job = steps::restart::scope(job);
    for w in &m.warnings {
        eprintln!("warning: {w}");
    }
    // *NODE PRINT/*EL PRINT/*ENERGY PRINT: every increment is needed for the .dat file (fem/dat.rs)
    let print = rustyfem::fem::dat::has_print(&m.model);
    let opts = RunOptions { all_frames: flag(rest, "--all-frames") || print, verbose: flag(rest, "-v"), solver: opt(rest, "--solver"), settings: None };
    let (out, _) = steps::run(&m.model, &opts)?;
    if print {
        let dat = path.with_extension("dat");
        std::fs::write(&dat, rustyfem::fem::dat::write(&m.model, &out))?;
        println!("Printout: {}", dat.display());
    }
    let rs = out.to_result_set(&m.model);
    println!("{}", path.display());
    println!("Nodes: {}   Elements: {}   Unknowns: {}", m.model.nodes.len(), m.model.elements.len(), out.n_eq);
    summarize(&rs);
    println!("Solution time: {:.3} s", t0.elapsed().as_secs_f64());
    if let Some(o) = opt(rest, "--out") {
        // the mesh as comment lines (#X...) for submodelling (steps/submodel.rs)
        std::fs::write(&o, rs.to_text() + &steps::submodel::mesh_text(&m.model))?;
        println!("Results: {o}");
    }
    Ok(true)
}

fn cmd_check(rest: &[String], update: bool) -> Result<bool, Box<dyn std::error::Error>> {
    let all = flag(rest, "--all");
    let positional: Vec<&String> = rest.iter().filter(|a| !a.starts_with("--")).collect();
    let files: Vec<PathBuf> = if all {
        let root = positional.first().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("tests/cases"));
        case::discover(&root)
    } else {
        positional.iter().map(PathBuf::from).collect()
    };
    if files.is_empty() {
        return Err("no regression cases (specify case.inp or --all; a case is an .inp with a .toml side file)".into());
    }
    let mut table = Vec::new();
    let mut all_ok = true;
    for f in &files {
        let name = case::case_name(f);
        println!("== {name} ==");
        match regression::check_case(f, update) {
            Ok(o) => {
                print!("{}", o.report.text());
                let ok = o.report.passed();
                all_ok &= ok;
                table.push((name, o.nodes, o.report.worst(), if ok { "OK" } else { "FAIL" }));
            }
            Err(e) => {
                println!("  FAIL {e}");
                all_ok = false;
                table.push((name, 0, f64::NAN, "FAIL"));
            }
        }
    }
    if files.len() > 1 {
        println!();
        println!("{:<44} {:>6} {:>12}  status", "case", "nodes", "max rel.err");
        for (n, nodes, w, s) in &table {
            println!("{:<44} {:>6} {:>12.2e}  {}", n, nodes, w, s);
        }
    }
    println!("{}", if all_ok { "All cases match the baseline within tolerance" } else { "DEVIATIONS FOUND" });
    Ok(all_ok)
}
