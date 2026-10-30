//! Results: displacements, reactions, stresses, section forces and export.

use crate::model::{Dim, Model};
use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Results {
    /// Displacement and rotation per node [ux,uy,uz,rx,ry,rz].
    pub disp: Vec<[f64; 6]>,
    /// Reaction forces/moments in constrained degrees of freedom.
    pub reaction: Vec<[f64; 6]>,
    /// Stresses per element and integration point [S11,S22,S33,S12,S13,S23].
    pub stress: Vec<Vec<[f64; 6]>>,
    /// Section forces in local coordinates for beam elements.
    pub beam_forces: Vec<Option<Vec<f64>>>,
    pub active: Vec<u8>,
    pub fixed: Vec<[bool; 6]>,
    /// Sum of the applied loads.
    pub applied: [f64; 6],
    /// Number of unknowns.
    pub n_eq: usize,
}

/// von Mises equivalent stress.
pub fn mises(s: &[f64; 6]) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2))
        + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
        .sqrt()
}

fn fmt(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 1e-3 && v.abs() < 1e5 {
        format!("{:.4}", v)
    } else {
        format!("{:.4e}", v)
    }
}

impl Results {
    /// (node, |u|) for the largest translation.
    pub fn max_displacement(&self) -> (usize, f64) {
        self.disp
            .iter()
            .enumerate()
            .map(|(i, u)| (i, (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()))
            .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a })
    }

    /// (element, von Mises) for the highest integration point value.
    pub fn max_mises(&self) -> Option<(usize, f64)> {
        self.stress
            .iter()
            .enumerate()
            .flat_map(|(e, ss)| ss.iter().map(move |s| (e, mises(s))))
            .fold(None, |a: Option<(usize, f64)>, b| match a {
                Some(a) if a.1 >= b.1 => Some(a),
                _ => Some(b),
            })
    }

    pub fn total_reaction(&self) -> [f64; 6] {
        let mut r = [0.0; 6];
        for rn in &self.reaction {
            for d in 0..6 {
                r[d] += rn[d];
            }
        }
        r
    }

    /// Averaged von Mises per node (element means distributed to the nodes).
    pub fn nodal_mises(&self, m: &Model) -> Vec<f64> {
        let mut sum = vec![0.0; m.nodes.len()];
        let mut cnt = vec![0usize; m.nodes.len()];
        for (e, ss) in self.stress.iter().enumerate() {
            if ss.is_empty() {
                continue;
            }
            let v = ss.iter().map(mises).sum::<f64>() / ss.len() as f64;
            for &n in &m.elements[e].nodes {
                sum[n] += v;
                cnt[n] += 1;
            }
        }
        sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
    }

    /// (element, |M|) for the largest bending moment in beams.
    pub fn max_beam_moment(&self, m: &Model) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (e, bf) in self.beam_forces.iter().enumerate() {
            let Some(f) = bf else { continue };
            let mv = if m.dim == Dim::Two {
                f[2].abs().max(f[5].abs())
            } else {
                [4, 5, 10, 11].iter().map(|&i| f[i].abs()).fold(0.0, f64::max)
            };
            if best.map_or(true, |b| mv > b.1) {
                best = Some((e, mv));
            }
        }
        best
    }

    pub fn summary(&self, m: &Model) -> String {
        let mut s = String::new();
        let n_nodes = m.nodes.len();
        let _ = writeln!(s, "Nodes: {n_nodes}   Elements: {}   Unknowns: {}", m.elements.len(), self.n_eq);
        let (n, umax) = self.max_displacement();
        let p = m.nodes[n];
        let u = self.disp[n];
        let _ = writeln!(
            s,
            "Max displacement:     {} at node {} at ({}, {}, {})  [ux={}, uy={}, uz={}]",
            fmt(umax),
            n + 1,
            fmt(p[0]),
            fmt(p[1]),
            fmt(p[2]),
            fmt(u[0]),
            fmt(u[1]),
            fmt(u[2])
        );
        if let Some((e, v)) = self.max_mises() {
            let _ = writeln!(s, "Max von Mises:        {} in element {}", fmt(v), e + 1);
        }
        if let Some((e, v)) = self.max_beam_moment(m) {
            let _ = writeln!(s, "Max beam moment:      {} in element {}", fmt(v), e + 1);
        }
        let r = self.total_reaction();
        let axes = if m.dim == Dim::Two { 2 } else { 3 };
        let names = ["x", "y", "z"];
        let rs: Vec<String> = (0..axes).map(|d| format!("R{}={}", names[d], fmt(r[d]))).collect();
        let _ = writeln!(s, "Total reactions:      {}", rs.join("  "));
        let scale = (0..3).map(|d| self.applied[d].abs()).fold(0.0, f64::max).max(1e-300);
        let resid = (0..3).map(|d| (r[d] + self.applied[d]).abs()).fold(0.0, f64::max) / scale;
        if (0..3).any(|d| self.applied[d] != 0.0) {
            let _ = writeln!(s, "Equilibrium check:    |ΣR + ΣF| / |F| = {:.1e}", resid);
        }
        s
    }

    /// Legacy VTK for ParaView: displacement, von Mises per node and element.
    pub fn write_vtk(&self, m: &Model, path: &Path) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "# vtk DataFile Version 3.0\nRustyFEM\nASCII\nDATASET UNSTRUCTURED_GRID")?;
        writeln!(f, "POINTS {} double", m.nodes.len())?;
        for p in &m.nodes {
            writeln!(f, "{:e} {:e} {:e}", p[0], p[1], p[2])?;
        }
        let size: usize = m.elements.iter().map(|e| e.nodes.len() + 1).sum();
        writeln!(f, "CELLS {} {}", m.elements.len(), size)?;
        for e in &m.elements {
            write!(f, "{}", e.nodes.len())?;
            for n in &e.nodes {
                write!(f, " {n}")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "CELL_TYPES {}", m.elements.len())?;
        for e in &m.elements {
            writeln!(f, "{}", m.elem_kind(e).vtk_type())?;
        }
        writeln!(f, "POINT_DATA {}\nVECTORS displacement double", m.nodes.len())?;
        for u in &self.disp {
            writeln!(f, "{:e} {:e} {:e}", u[0], u[1], u[2])?;
        }
        writeln!(f, "SCALARS mises double 1\nLOOKUP_TABLE default")?;
        for v in self.nodal_mises(m) {
            writeln!(f, "{v:e}")?;
        }
        writeln!(f, "CELL_DATA {}\nSCALARS mises_element double 1\nLOOKUP_TABLE default", m.elements.len())?;
        for ss in &self.stress {
            let v = if ss.is_empty() { 0.0 } else { ss.iter().map(mises).sum::<f64>() / ss.len() as f64 };
            writeln!(f, "{v:e}")?;
        }
        Ok(())
    }

    /// Node results as CSV.
    pub fn write_csv(&self, m: &Model, path: &Path) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "node,x,y,z,ux,uy,uz,rx,ry,rz,rfx,rfy,rfz,rmx,rmy,rmz,mises")?;
        let mis = self.nodal_mises(m);
        for (i, p) in m.nodes.iter().enumerate() {
            let u = self.disp[i];
            let r = self.reaction[i];
            writeln!(
                f,
                "{},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e},{:e}",
                i + 1,
                p[0],
                p[1],
                p[2],
                u[0],
                u[1],
                u[2],
                u[3],
                u[4],
                u[5],
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                mis[i]
            )?;
        }
        Ok(())
    }
}
