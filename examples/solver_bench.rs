//! Benchmark of the linear solvers on generated models.
//!
//! cargo run --release --example solver_bench [-- n1 n2 ...]
//!
//! 1. C3D8 cube with n×n×n elements (fixed bottom, loads on the top face): the whole
//!    analysis (`steps::run`) with each solver; time and largest relative
//!    deviation in U from `direct`.
//! 2. Indefinite: 7-point Laplacian on (2n)³ nodes shifted by σ so that about
//!    10 % of the eigenvalues become negative; time for `sparse::indef` and inertia
//!    compared with the analytical one.

use rustyfem::sparse::{self, indef};
use rustyfem::steps::{self, RunOptions};
use std::path::Path;
use std::time::Instant;

fn cube(n: usize) -> String {
    let id = |i: usize, j: usize, k: usize| 1 + i + j * (n + 1) + k * (n + 1) * (n + 1);
    let mut s = String::from("*HEADING\ncube\n*NODE\n");
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                s += &format!("{}, {}, {}, {}\n", id(i, j, k), i as f64 / n as f64, j as f64 / n as f64, k as f64 / n as f64);
            }
        }
    }
    s += "*ELEMENT, TYPE=C3D8, ELSET=ALLA\n";
    let mut e = 1;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                s += &format!(
                    "{e}, {}, {}, {}, {}, {}, {}, {}, {}\n",
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1)
                );
                e += 1;
            }
        }
    }
    s += &format!("*NSET, NSET=BOTTEN, GENERATE\n1, {}, 1\n", (n + 1) * (n + 1));
    s += &format!("*ELSET, ELSET=TOPP, GENERATE\n{}, {}, 1\n", n * n * (n - 1) + 1, n * n * n);
    s += "*SOLID SECTION, ELSET=ALLA, MATERIAL=M\n,\n*MATERIAL, NAME=M\n*ELASTIC\n2.1e11, 0.3\n";
    s += "*BOUNDARY\nBOTTEN, ENCASTRE\n*STEP\n*STATIC\n*DLOAD\nTOPP, P6, 1e6\n";
    s += &format!("*CLOAD\n{}, 1, 1e5\n*END STEP\n", id(n, n, n));
    s
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let sizes = if args.is_empty() { vec![10, 20, 30] } else { args };
    println!("threads: {}", std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1));
    println!("\n| model | equations | solver | analysis time [s] | max rel. deviation U vs direct |");
    println!("|---|---:|---|---:|---:|");
    for &n in &sizes {
        let text = cube(n);
        let m = rustyfem::inp::read_str(&text, Path::new(".")).expect("inp");
        let mut reference: Option<Vec<f64>> = None;
        for s in ["direct", "indefinite", "unsymmetric", "iterative", "iterative-ic0", "iterative-jacobi"] {
            if s == "iterative-jacobi" && n > 30 {
                continue;
            }
            let t0 = Instant::now();
            let (out, st) = steps::run(&m.model, &RunOptions { solver: Some(s.into()), ..Default::default() }).expect("analysis");
            let dt = t0.elapsed().as_secs_f64();
            let u: Vec<f64> = st.u.iter().flat_map(|u| u[..3].to_vec()).collect();
            let err = match &reference {
                None => {
                    reference = Some(u);
                    0.0
                }
                Some(r) => {
                    let sc = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                    u.iter().zip(r).fold(0.0f64, |a, (x, y)| a.max((x - y).abs())) / sc
                }
            };
            println!("| C3D8 {n}³ | {} | {s} | {dt:.2} | {err:.1e} |", out.n_eq);
        }
    }

    println!("\n| Laplace (indefinite) | equations | σ | factorization [s] | negative pivots | analytical | delayed | 2×2 |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for &n in &sizes {
        let g = 2 * n;
        let nn = g * g * g;
        let id = |i: usize, j: usize, k: usize| i + j * g + k * g * g;
        let mut coords = Vec::with_capacity(nn);
        let mut adj = vec![Vec::new(); nn];
        for k in 0..g {
            for j in 0..g {
                for i in 0..g {
                    coords.push([i as f64, j as f64, k as f64]);
                    let v = id(i, j, k);
                    if i > 0 {
                        adj[v].push(id(i - 1, j, k));
                    }
                    if i + 1 < g {
                        adj[v].push(id(i + 1, j, k));
                    }
                    if j > 0 {
                        adj[v].push(id(i, j - 1, k));
                    }
                    if j + 1 < g {
                        adj[v].push(id(i, j + 1, k));
                    }
                    if k > 0 {
                        adj[v].push(id(i, j, k - 1));
                    }
                    if k + 1 < g {
                        adj[v].push(id(i, j, k + 1));
                    }
                    adj[v].sort_unstable();
                }
            }
        }
        let ndof = vec![1usize; nn];
        let sym = sparse::analyse(&coords, &adj, &ndof);
        // eigenvalues 6 − 2Σcos(k_a π/(g+1)); σ at about the 10 % quantile
        let c: Vec<f64> = (1..=g).map(|k| 2.0 * (k as f64 * std::f64::consts::PI / (g + 1) as f64).cos()).collect();
        let mut lam = Vec::with_capacity(nn);
        for a in &c {
            for b in &c {
                for d in &c {
                    lam.push(6.0 - a - b - d);
                }
            }
        }
        lam.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // midway between two distinct eigenvalues (the eigenvalues have high multiplicity)
        let mut q = nn / 10;
        while lam[q + 1] - lam[q] < 1e-9 {
            q += 1;
        }
        let sigma = 0.5 * (lam[q] + lam[q + 1]);
        let below = lam.iter().filter(|&&l| l < sigma).count();
        let mut k = sym.pattern(&adj, &ndof);
        for v in 0..nn {
            let q = sym.first_dof[v];
            k.add(q, q, 6.0 - sigma);
            for &w in &adj[v] {
                let qw = sym.first_dof[w];
                if qw > q {
                    k.add(qw, q, -1.0);
                }
            }
        }
        let t0 = Instant::now();
        let f = indef::factor(&sym, &k).expect("indefinite factorization");
        let dt = t0.elapsed().as_secs_f64();
        println!("| {g}³ | {nn} | {sigma:.4} | {dt:.2} | {} | {below} | {} | {} |", f.inertia.neg, f.delayed, f.two_by_two);
    }
}
