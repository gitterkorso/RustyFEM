use rustyfem::fem::output::ResultSet;
use rustyfem::results::mises;
use rustyfem::steps::{self, RunOptions};
use rustyfem::regression::{self, case};
use rustyfem::{inp, script};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const HELP: &str = "RustyFEM – FEM-lösare för .rusty-skript och importerade .inp-indatafiler

Användning:
  rustyfem <modell.rusty|modell.inp>        kör analysen
  rustyfem run <modell.rusty> [--vtk F] [--csv F]
  rustyfem run <modell.inp> [--out F] [--all-frames] [-v]
                                           --out skriver alla resultat i textformat
                    [--oldjob jobb] [--global global.res]
                                           --oldjob: omstartfil för *RESTART, READ (jobb.rfres)
                                           --global: globalmodellens resultat för *SUBMODEL
                                           (--out-fil eller .rfres)
  rustyfem check <fall.inp ...>             kör regressionsfall och jämför med fall.ref
  rustyfem check --all [katalog]            alla fall under tests/cases/ i testmappen (eller katalog)
  rustyfem baseline <fall.inp ...|--all [katalog]>
                                           skriver om fall.ref från RustyFEM
  rustyfem list                             registrerade element, nyckelord, steg och lösare

Miljövariabler: RUSTYFEM_SOLVER (linjär lösare),
RUSTYFEM_USER (delat bibliotek med umat_/uel_ för *USER MATERIAL/*USER ELEMENT).";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("fel: {e}");
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
            let f = args.get(1).ok_or("ange en modellfil")?;
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
    println!("Element:    {}", rustyfem::elements::names().join(", "));
    println!("Nyckelord:  {}", inp::names().iter().map(|k| format!("*{k}")).collect::<Vec<_>>().join(", "));
    println!("Steg-typer: {}", steps::all().iter().map(|s| s.name()).collect::<Vec<_>>().join(", "));
    println!("Material:   {}", rustyfem::materials::factories().iter().map(|f| f.name).collect::<Vec<_>>().join(", "));
    println!("Lösare:     {}", rustyfem::solver::all().iter().map(|s| s.name()).collect::<Vec<_>>().join(", "));
}

fn cmd_run(file: &str, rest: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let path = PathBuf::from(file);
    let model = script::load(&path)?;
    let t0 = std::time::Instant::now();
    let res = model.solve()?;
    println!("{}", path.display());
    print!("{}", res.summary(&model));
    println!("Lösningstid:          {:.3} s", t0.elapsed().as_secs_f64());
    let vtk = opt(rest, "--vtk").map(PathBuf::from).unwrap_or_else(|| path.with_extension("vtk"));
    res.write_vtk(&model, &vtk)?;
    println!("Resultat för ParaView: {}", vtk.display());
    if let Some(csv) = opt(rest, "--csv") {
        res.write_csv(&model, Path::new(&csv))?;
        println!("Nodresultat:          {csv}");
    }
    Ok(true)
}

/// Kort sammanfattning av sista framen i varje steg.
fn summarize(rs: &ResultSet) {
    for st in &rs.steps {
        println!("Steg {} ({}, {}): {} frames", st.index, st.name, st.procedure, st.frames.len());
        let Some(fr) = st.frames.last() else { continue };
        if let Some(u) = fr.node.get("U") {
            let best = u.items.iter().map(|(k, c)| (k, c.iter().map(|p| p.1 * p.1).sum::<f64>().sqrt())).fold(None, |a: Option<(&(String, i64), f64)>, b| match a {
                Some(a) if a.1 >= b.1 => Some(a),
                _ => Some(b),
            });
            if let Some(((inst, n), v)) = best {
                println!("  största förskjutning |U| = {v:.6e} i nod {inst}.{n}");
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
            println!("  summa reaktioner: {}", parts.join("  "));
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
            println!("  högsta von Mises = {:.6e} i element {}", best.0, best.1);
        }
    }
}

fn cmd_run_inp(file: &str, rest: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let path = PathBuf::from(file);
    let t0 = std::time::Instant::now();
    // jobbkontext: omstartfil, matriser och substrukturer skrivs bredvid indatafilen
    let mut job = steps::restart::Job::for_file(&path);
    job.oldjob = opt(rest, "--oldjob").map(PathBuf::from);
    job.global = opt(rest, "--global").map(PathBuf::from);
    let m = steps::restart::read_job(&path, &mut job)?;
    let _job = steps::restart::scope(job);
    for w in &m.warnings {
        eprintln!("varning: {w}");
    }
    // *NODE PRINT/*EL PRINT/*ENERGY PRINT: alla inkrement behövs för .dat-filen (fem/dat.rs)
    let print = rustyfem::fem::dat::has_print(&m.model);
    let opts = RunOptions { all_frames: flag(rest, "--all-frames") || print, verbose: flag(rest, "-v"), solver: opt(rest, "--solver"), settings: None };
    let (out, _) = steps::run(&m.model, &opts)?;
    if print {
        let dat = path.with_extension("dat");
        std::fs::write(&dat, rustyfem::fem::dat::write(&m.model, &out))?;
        println!("Utskrift: {}", dat.display());
    }
    let rs = out.to_result_set(&m.model);
    println!("{}", path.display());
    println!("Noder: {}   Element: {}   Obekanta: {}", m.model.nodes.len(), m.model.elements.len(), out.n_eq);
    summarize(&rs);
    println!("Lösningstid: {:.3} s", t0.elapsed().as_secs_f64());
    if let Some(o) = opt(rest, "--out") {
        // nätet som kommentarsrader (#X...) för submodellering (steps/submodel.rs)
        std::fs::write(&o, rs.to_text() + &steps::submodel::mesh_text(&m.model))?;
        println!("Resultat: {o}");
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
        return Err("inga regressionsfall (ange fall.inp eller --all; ett fall är en .inp med en .toml-sidofil)".into());
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
                table.push((name, o.nodes, o.report.worst(), if ok { "OK" } else { "FEL" }));
            }
            Err(e) => {
                println!("  FEL {e}");
                all_ok = false;
                table.push((name, 0, f64::NAN, "FEL"));
            }
        }
    }
    if files.len() > 1 {
        println!();
        println!("{:<44} {:>6} {:>12}  status", "fall", "noder", "max rel.fel");
        for (n, nodes, w, s) in &table {
            println!("{:<44} {:>6} {:>12.2e}  {}", n, nodes, w, s);
        }
    }
    println!("{}", if all_ok { "Alla fall stämmer med baslinjen inom toleransen" } else { "AVVIKELSER FINNS" });
    Ok(all_ok)
}
