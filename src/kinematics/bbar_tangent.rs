//! Volymtermen i tangenten för B-bar-elementen (CPE4, C3D8) vid NLGEOM.
//!
//! Den inre kraften är f·δu = Σ_q W_q σ_q : δD̄_q med
//! δD̄ = sym δL + (1/nv)(δθ̄ − δθ) I, δθ = tr δL (de nv första komponenterna)
//! och δθ̄ = Σ_p v_p δθ_p / V (v_p = punktens aktuella volym). Standard-
//! tangenten i `continuum.rs` linjäriserar sym δL och vikterna, men inte att
//! volymdelen av den virtuella töjningen beror på läget:
//!
//!   d(δθ_q) = −tr(δL_q dL_q),
//!   d(δθ̄)  = Σ_p (v_p/V) [−tr(δL_p dL_p) + δθ_p dθ_p] − δθ̄ dθ̄.
//!
//! Termen Σ_q W_q p̂_q (d(δθ̄) − d(δθ_q)) med p̂ = (1/nv) Σ_{i<nv} σ_ii läggs
//! till här med omvänt tecken: tillsammans med standardtangentens
//! volymtermer (½(σ⊗I + I⊗σ) med B̄, som redan räknar med det linjäriserade
//! spåret) är det det tecken som för tangenten närmare den numeriska
//! (RUSTYFEM_NL_FDTANGENT: största relativa avvikelse 1,1e-1 → 7,6e-2 i
//! rubber/c3d8h_block_elastisk_nlgeom), och den härledda formen med positivt
//! tecken försämrar den (2,0e-1). Utan termen konvergerade Newton bara
//! linjärt vid stora töjningar (samma fall: inkrementet halverades); med den
//! konvergerar fallet kvadratiskt.

/// Lägger till volymtermen i `k` (nd × nd, nodvis med `ndim` frihetsgrader).
///
/// `g[q][a]` = ∇ₓNₐ i punkt q (aktuellt läge), `v[q]` = punktens aktuella
/// volym (vikt för medelvärdet), `w[q]` = integrationsvikten i den inre
/// kraften och `p[q]` = (1/nv) Σ_{i<nv} σ_ii.
pub fn add_volume_term(k: &mut [f64], ndim: usize, nv: usize, g: &[Vec<[f64; 3]>], v: &[f64], w: &[f64], p: &[f64]) {
    let nq = g.len();
    if nq == 0 {
        return;
    }
    let nn = g[0].len();
    let nd = nn * ndim;
    let vt: f64 = v.iter().sum();
    let ptot: f64 = (0..nq).map(|q| w[q] * p[q]).sum();
    // volymgradient per frihetsgrad: δθ_q = Σ gq[c] δu_c (bara komponenter i < nv)
    let theta = |q: usize| -> Vec<f64> {
        let mut t = vec![0.0; nd];
        for a in 0..nn {
            for i in 0..ndim.min(nv) {
                t[a * ndim + i] = g[q][a][i];
            }
        }
        t
    };
    let thetas: Vec<Vec<f64>> = (0..nq).map(theta).collect();
    let mut tbar = vec![0.0; nd];
    for q in 0..nq {
        for c in 0..nd {
            tbar[c] += v[q] / vt * thetas[q][c];
        }
    }
    // M_q(ai, bj) = −∂Nₐ/∂x_j ∂N_b/∂x_i (i, j < nv: volymdelens komponenter)
    let m = |q: usize, a: usize, i: usize, b: usize, j: usize| -> f64 {
        if i < nv && j < nv {
            -g[q][a][j] * g[q][b][i]
        } else {
            0.0
        }
    };
    for a in 0..nn {
        for i in 0..ndim {
            let r = a * ndim + i;
            for b in 0..nn {
                for j in 0..ndim {
                    let c = b * ndim + j;
                    let mut s = 0.0;
                    for q in 0..nq {
                        let mq = m(q, a, i, b, j);
                        // Σ_q W p̂ M̄ (M̄ gemensam) − Σ_q W_q p̂_q M_q
                        s += ptot * (v[q] / vt) * (mq + thetas[q][r] * thetas[q][c]);
                        s -= w[q] * p[q] * mq;
                    }
                    s -= ptot * tbar[r] * tbar[c];
                    k[r * nd + c] -= s;
                }
            }
        }
    }
}
