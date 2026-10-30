//! Geometri för XFEM: referenselement, uppdelning av element längs
//! nivåmängdsfunktionerna, kvadraturregler på delsimplex och sprickspetsens
//! asymptotiska funktioner.
//!
//! Allt arbetar i elementets naturliga koordinater. Ett element delas först i
//! simplex (hexaedern i sex tetraedrar runt diagonalen nod 1–7, fyrhörningen i
//! två trianglar). Varje simplex skärs sedan av den linjära interpolationen av
//! φ (sprickytan) och, i spetselement, av ψ (fronten). Delsimplexen får en
//! sida (tecknet på φ i tyngdpunkten) och integreras med kollapsade
//! gaussregler (Stroud), "internt genererade integrationspunkter".

/// Referenselementets typ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    Hex8,
    Tet4,
    Quad4,
    Tri3,
}

const HEX_NAT: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

impl Ref {
    /// Referenselement för ett elementnamn (de element som får berikas).
    pub fn of(name: &str) -> Option<Ref> {
        match name {
            "C3D8" | "C3D8R" => Some(Ref::Hex8),
            "C3D4" => Some(Ref::Tet4),
            "CPE4" | "CPS4" | "CPE4R" | "CPS4R" => Some(Ref::Quad4),
            "CPE3" | "CPS3" => Some(Ref::Tri3),
            _ => None,
        }
    }

    pub fn dim(self) -> usize {
        match self {
            Ref::Hex8 | Ref::Tet4 => 3,
            Ref::Quad4 | Ref::Tri3 => 2,
        }
    }

    pub fn nn(self) -> usize {
        match self {
            Ref::Hex8 => 8,
            Ref::Tet4 | Ref::Quad4 => 4,
            Ref::Tri3 => 3,
        }
    }

    /// Nodernas naturliga koordinater.
    pub fn vertices(self) -> Vec<[f64; 3]> {
        match self {
            Ref::Hex8 => HEX_NAT.to_vec(),
            Ref::Tet4 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            Ref::Quad4 => vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            Ref::Tri3 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    /// Tyngdpunkten i naturliga koordinater.
    pub fn centroid(self) -> [f64; 3] {
        match self {
            Ref::Hex8 | Ref::Quad4 => [0.0; 3],
            Ref::Tet4 => [0.25, 0.25, 0.25],
            Ref::Tri3 => [1.0 / 3.0, 1.0 / 3.0, 0.0],
        }
    }

    /// Uppdelning i simplex (nodindex).
    pub fn simplices(self) -> Vec<Vec<usize>> {
        match self {
            Ref::Hex8 => vec![vec![0, 1, 2, 6], vec![0, 2, 3, 6], vec![0, 3, 7, 6], vec![0, 7, 4, 6], vec![0, 4, 5, 6], vec![0, 5, 1, 6]],
            Ref::Tet4 => vec![vec![0, 1, 2, 3]],
            Ref::Quad4 => vec![vec![0, 1, 2], vec![0, 2, 3]],
            Ref::Tri3 => vec![vec![0, 1, 2]],
        }
    }

    /// Elementets ytor (3D) respektive kanter (2D) som nodindex, i .inp-formatets ordning.
    pub fn faces(self) -> Vec<Vec<usize>> {
        match self {
            Ref::Hex8 => vec![vec![0, 1, 2, 3], vec![4, 7, 6, 5], vec![0, 4, 5, 1], vec![1, 5, 6, 2], vec![2, 6, 7, 3], vec![3, 7, 4, 0]],
            Ref::Tet4 => vec![vec![0, 1, 2], vec![0, 3, 1], vec![1, 3, 2], vec![2, 3, 0]],
            Ref::Quad4 => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
            Ref::Tri3 => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
        }
    }

    /// Formfunktioner och derivator mot de naturliga koordinaterna.
    pub fn shape(self, p: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
        match self {
            Ref::Hex8 => {
                let mut n = vec![0.0; 8];
                let mut d = vec![[0.0; 3]; 8];
                for i in 0..8 {
                    let a = HEX_NAT[i];
                    let f = [1.0 + p[0] * a[0], 1.0 + p[1] * a[1], 1.0 + p[2] * a[2]];
                    n[i] = 0.125 * f[0] * f[1] * f[2];
                    d[i] = [0.125 * a[0] * f[1] * f[2], 0.125 * a[1] * f[0] * f[2], 0.125 * a[2] * f[0] * f[1]];
                }
                (n, d)
            }
            Ref::Tet4 => (
                vec![1.0 - p[0] - p[1] - p[2], p[0], p[1], p[2]],
                vec![[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            Ref::Quad4 => {
                let a = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                let mut n = vec![0.0; 4];
                let mut d = vec![[0.0; 3]; 4];
                for i in 0..4 {
                    n[i] = 0.25 * (1.0 + a[i][0] * p[0]) * (1.0 + a[i][1] * p[1]);
                    d[i] = [0.25 * a[i][0] * (1.0 + a[i][1] * p[1]), 0.25 * a[i][1] * (1.0 + a[i][0] * p[0]), 0.0];
                }
                (n, d)
            }
            Ref::Tri3 => (vec![1.0 - p[0] - p[1], p[0], p[1]], vec![[-1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        }
    }

    /// Gaussregel över hela referenselementet (n punkter per riktning för
    /// hexaeder/fyrhörning, kollapsad regel för simplex). Vikterna summerar
    /// till referenselementets mått.
    pub fn rule(self, n: usize) -> Vec<([f64; 3], f64)> {
        let g = gauss_m11(n);
        match self {
            Ref::Hex8 => {
                let mut v = Vec::new();
                for &(z, wz) in &g {
                    for &(y, wy) in &g {
                        for &(x, wx) in &g {
                            v.push(([x, y, z], wx * wy * wz));
                        }
                    }
                }
                v
            }
            Ref::Quad4 => {
                let mut v = Vec::new();
                for &(y, wy) in &g {
                    for &(x, wx) in &g {
                        v.push(([x, y, 0.0], wx * wy));
                    }
                }
                v
            }
            Ref::Tet4 | Ref::Tri3 => {
                let verts = self.vertices();
                let m = if self == Ref::Tet4 { 1.0 / 6.0 } else { 0.5 };
                simplex_rule(self.dim(), n).into_iter().map(|(b, w)| (bary_point(&verts, &b), w * m)).collect()
            }
        }
    }
}

/// Gauss–Legendre på [−1, 1] med n punkter (1–5).
pub fn gauss_m11(n: usize) -> Vec<(f64, f64)> {
    match n {
        0 | 1 => vec![(0.0, 2.0)],
        2 => {
            let a = 0.577_350_269_189_625_8;
            vec![(-a, 1.0), (a, 1.0)]
        }
        3 => {
            let a = 0.774_596_669_241_483_4;
            vec![(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
        }
        4 => {
            let (a, b) = (0.339_981_043_584_856_3, 0.861_136_311_594_052_6);
            let (wa, wb) = (0.652_145_154_862_546_1, 0.347_854_845_137_453_9);
            vec![(-b, wb), (-a, wa), (a, wa), (b, wb)]
        }
        _ => {
            let (a, b) = (0.538_469_310_105_683_1, 0.906_179_845_938_664);
            let (w0, wa, wb) = (0.568_888_888_888_888_9, 0.478_628_670_499_366_5, 0.236_926_885_056_189_1);
            vec![(-b, wb), (-a, wa), (0.0, w0), (a, wa), (b, wb)]
        }
    }
}

/// Kvadratur på ett simplex i barycentriska koordinater; vikterna summerar till 1.
/// Ordning 1: tyngdpunkten, 2: de klassiska 3- respektive 4-punktsreglerna,
/// högre: kollapsad Gauss med `order` punkter per riktning.
pub fn simplex_rule(dim: usize, order: usize) -> Vec<(Vec<f64>, f64)> {
    match (dim, order) {
        (2, 0 | 1) => vec![(vec![1.0 / 3.0; 3], 1.0)],
        (3, 0 | 1) => vec![(vec![0.25; 4], 1.0)],
        (2, 2) => {
            let (a, b) = (2.0 / 3.0, 1.0 / 6.0);
            vec![(vec![a, b, b], 1.0 / 3.0), (vec![b, a, b], 1.0 / 3.0), (vec![b, b, a], 1.0 / 3.0)]
        }
        (3, 2) => {
            let (a, b) = (0.585_410_196_624_968_5, 0.138_196_601_125_010_5);
            (0..4).map(|k| ((0..4).map(|i| if i == k { a } else { b }).collect(), 0.25)).collect()
        }
        _ => {
            let g: Vec<(f64, f64)> = gauss_m11(order).into_iter().map(|(x, w)| (0.5 * (x + 1.0), 0.5 * w)).collect();
            let mut v = Vec::new();
            if dim == 2 {
                for &(s, ws) in &g {
                    for &(t, wt) in &g {
                        let l1 = s;
                        let l2 = t * (1.0 - s);
                        v.push((vec![1.0 - l1 - l2, l1, l2], 2.0 * ws * wt * (1.0 - s)));
                    }
                }
            } else {
                for &(s, ws) in &g {
                    for &(t, wt) in &g {
                        for &(r, wr) in &g {
                            let l1 = s;
                            let l2 = t * (1.0 - s);
                            let l3 = r * (1.0 - s) * (1.0 - t);
                            v.push((vec![1.0 - l1 - l2 - l3, l1, l2, l3], 6.0 * ws * wt * wr * (1.0 - s) * (1.0 - s) * (1.0 - t)));
                        }
                    }
                }
            }
            v
        }
    }
}

pub fn bary_point(v: &[[f64; 3]], b: &[f64]) -> [f64; 3] {
    let mut p = [0.0; 3];
    for (vi, bi) in v.iter().zip(b) {
        for k in 0..3 {
            p[k] += bi * vi[k];
        }
    }
    p
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Simplexets mått (area eller volym) i de koordinater hörnen anges i.
pub fn measure(v: &[[f64; 3]]) -> f64 {
    if v.len() == 3 {
        let (a, b) = (sub(v[1], v[0]), sub(v[2], v[0]));
        0.5 * (a[0] * b[1] - a[1] * b[0]).abs()
    } else {
        let (a, b, c) = (sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]));
        (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])).abs() / 6.0
    }
}

/// Ett delsimplex (hörn i naturliga koordinater).
#[derive(Debug, Clone)]
pub struct Cell {
    pub v: Vec<[f64; 3]>,
}

/// Skär ett simplex med den linjära funktionen med hörnvärdena `f`.
/// Returnerar delarna med f ≥ 0, delarna med f < 0 och snittytan
/// (segment i 2D, triangel/fyrhörning som trianglar i 3D).
pub fn cut_simplex(v: &[[f64; 3]], f: &[f64]) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>, Vec<Vec<[f64; 3]>>) {
    let pos: Vec<usize> = (0..v.len()).filter(|&i| f[i] >= 0.0).collect();
    let neg: Vec<usize> = (0..v.len()).filter(|&i| f[i] < 0.0).collect();
    if neg.is_empty() {
        return (vec![v.to_vec()], Vec::new(), Vec::new());
    }
    if pos.is_empty() {
        return (Vec::new(), vec![v.to_vec()], Vec::new());
    }
    let p = |a: usize, b: usize| -> [f64; 3] {
        let t = f[a] / (f[a] - f[b]);
        lerp(v[a], v[b], t.clamp(0.0, 1.0))
    };
    if v.len() == 3 {
        // triangel: en ensam hörnpunkt mot två
        let (lone, others, lone_pos) = if pos.len() == 1 { (pos[0], &neg, true) } else { (neg[0], &pos, false) };
        let (b, c) = (others[0], others[1]);
        let (pb, pc) = (p(lone, b), p(lone, c));
        let tri = vec![v[lone], pb, pc];
        let quad = vec![vec![pb, v[b], v[c]], vec![pb, v[c], pc]];
        let seg = vec![vec![pb, pc]];
        return if lone_pos { (vec![tri], quad, seg) } else { (quad, vec![tri], seg) };
    }
    // tetraeder
    let prism = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| -> Vec<Vec<[f64; 3]>> {
        vec![vec![a[0], a[1], a[2], b[0]], vec![a[1], a[2], b[0], b[1]], vec![a[2], b[0], b[1], b[2]]]
    };
    if pos.len() == 1 || neg.len() == 1 {
        let (lone, others, lone_pos) = if pos.len() == 1 { (pos[0], &neg, true) } else { (neg[0], &pos, false) };
        let (b, c, d) = (others[0], others[1], others[2]);
        let (pb, pc, pd) = (p(lone, b), p(lone, c), p(lone, d));
        let tet = vec![v[lone], pb, pc, pd];
        let rest = prism([pb, pc, pd], [v[b], v[c], v[d]]);
        let face = vec![vec![pb, pc, pd]];
        return if lone_pos { (vec![tet], rest, face) } else { (rest, vec![tet], face) };
    }
    let (a, b) = (pos[0], pos[1]);
    let (c, d) = (neg[0], neg[1]);
    let (pac, pad, pbc, pbd) = (p(a, c), p(a, d), p(b, c), p(b, d));
    let side_pos = prism([v[a], pac, pad], [v[b], pbc, pbd]);
    let side_neg = prism([v[c], pac, pbc], [v[d], pad, pbd]);
    let face = vec![vec![pac, pad, pbd], vec![pac, pbd, pbc]];
    (side_pos, side_neg, face)
}

/// Interpolerar nodvärdena `vals` i den naturliga punkten `p`.
pub fn interp(r: Ref, vals: &[f64], p: [f64; 3]) -> f64 {
    let (n, _) = r.shape(p);
    n.iter().zip(vals).map(|(a, b)| a * b).sum()
}

/// Delceller i ett element med sida (+1/−1 efter φ) och snittytor (φ = 0).
/// `psi` = Some(...) delar också längs ψ = 0 (spetselement).
pub struct Partition {
    pub cells: Vec<(Cell, i8)>,
    pub facets: Vec<Vec<[f64; 3]>>,
}

/// Delar referenselementet längs φ = 0 (och ψ = 0 om `psi` ges).
pub fn partition(r: Ref, phi: &[f64], psi: Option<&[f64]>) -> Partition {
    let verts = r.vertices();
    let mut cells: Vec<Vec<[f64; 3]>> = r.simplices().into_iter().map(|s| s.iter().map(|&i| verts[i]).collect()).collect();
    if let Some(ps) = psi {
        let mut next = Vec::new();
        for c in &cells {
            let f: Vec<f64> = c.iter().map(|&p| interp(r, ps, p)).collect();
            let (a, b, _) = cut_simplex(c, &f);
            next.extend(a);
            next.extend(b);
        }
        cells = next;
    }
    let mut out = Vec::new();
    let mut facets = Vec::new();
    for c in &cells {
        let f: Vec<f64> = c.iter().map(|&p| interp(r, phi, p)).collect();
        let (a, b, fa) = cut_simplex(c, &f);
        for x in a {
            if measure(&x) > 1e-14 {
                out.push((Cell { v: x }, 1));
            }
        }
        for x in b {
            if measure(&x) > 1e-14 {
                out.push((Cell { v: x }, -1));
            }
        }
        // snittytan bara där sprickan finns (ψ < 0)
        for x in fa {
            let keep = match psi {
                Some(ps) => {
                    let c: [f64; 3] = bary_point(&x, &vec![1.0 / x.len() as f64; x.len()]);
                    interp(r, ps, c) < 0.0
                }
                None => true,
            };
            if keep {
                facets.push(x);
            }
        }
    }
    Partition { cells: out, facets }
}

/// Sprickspetsens fyra asymptotiska funktioner och deras gradienter för
/// nivåmängdsvärdena (φ, ψ) och gradienterna (∇φ, ∇ψ):
/// F = √r [sin θ/2, cos θ/2, sin θ/2 sin θ, cos θ/2 sin θ], r = √(φ² + ψ²), θ = atan2(φ, ψ).
pub fn tip_functions(phi: f64, psi: f64, gphi: [f64; 3], gpsi: [f64; 3]) -> ([f64; 4], [[f64; 3]; 4]) {
    let r2 = (phi * phi + psi * psi).max(1e-300);
    let r = r2.sqrt();
    let th = phi.atan2(psi);
    let sr = r.sqrt();
    let (s2, c2) = (0.5 * th).sin_cos();
    let (s, c) = th.sin_cos();
    let f = [sr * s2, sr * c2, sr * s2 * s, sr * c2 * s];
    // derivator mot r och θ
    let fr = [s2 / (2.0 * sr), c2 / (2.0 * sr), s2 * s / (2.0 * sr), c2 * s / (2.0 * sr)];
    let ft = [0.5 * sr * c2, -0.5 * sr * s2, sr * (0.5 * c2 * s + s2 * c), sr * (-0.5 * s2 * s + c2 * c)];
    let mut g = [[0.0; 3]; 4];
    for k in 0..3 {
        let rx = (phi * gphi[k] + psi * gpsi[k]) / r;
        let tx = (psi * gphi[k] - phi * gpsi[k]) / r2;
        for a in 0..4 {
            g[a][k] = fr[a] * rx + ft[a] * tx;
        }
    }
    (f, g)
}

/// Värdena av de asymptotiska funktionerna i en nod (för de förskjutna funktionerna).
pub fn tip_values(phi: f64, psi: f64) -> [f64; 4] {
    tip_functions(phi, psi, [0.0; 3], [0.0; 3]).0
}
