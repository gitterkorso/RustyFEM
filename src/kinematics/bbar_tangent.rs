//! The volumetric term in the tangent for the B-bar elements (CPE4, C3D8) with NLGEOM.
//!
//! The internal force is f·δu = Σ_q W_q σ_q : δD̄_q with
//! δD̄ = sym δL + (1/nv)(δθ̄ − δθ) I, δθ = tr δL (the first nv components)
//! and δθ̄ = Σ_p v_p δθ_p / V (v_p = current volume of the point). The standard
//! tangent in `continuum.rs` linearizes sym δL and the weights, but not that
//! the volumetric part of the virtual strain depends on the configuration:
//!
//!   d(δθ_q) = −tr(δL_q dL_q),
//!   d(δθ̄)  = Σ_p (v_p/V) [−tr(δL_p dL_p) + δθ_p dθ_p] − δθ̄ dθ̄.
//!
//! The term Σ_q W_q p̂_q (d(δθ̄) − d(δθ_q)) with p̂ = (1/nv) Σ_{i<nv} σ_ii is added
//! here with the opposite sign: together with the standard tangent's
//! volumetric terms (½(σ⊗I + I⊗σ) with B̄, which already account for the linearized
//! trace) this is the sign that brings the tangent closer to the numerical one
//! (RUSTYFEM_NL_FDTANGENT: largest relative deviation 1.1e-1 → 7.6e-2 in
//! rubber/c3d8h_block_elastisk_nlgeom), and the derived form with positive
//! sign makes it worse (2.0e-1). Without the term Newton converged only
//! linearly at large strains (same case: the increment was halved); with it
//! the case converges quadratically.

/// Adds the volumetric term to `k` (nd × nd, node by node with `ndim` degrees of freedom).
///
/// `g[q][a]` = ∇ₓNₐ at point q (current configuration), `v[q]` = current volume
/// of the point (weight for the mean), `w[q]` = the integration weight in the internal
/// force and `p[q]` = (1/nv) Σ_{i<nv} σ_ii.
pub fn add_volume_term(k: &mut [f64], ndim: usize, nv: usize, g: &[Vec<[f64; 3]>], v: &[f64], w: &[f64], p: &[f64]) {
    let nq = g.len();
    if nq == 0 {
        return;
    }
    let nn = g[0].len();
    let nd = nn * ndim;
    let vt: f64 = v.iter().sum();
    let ptot: f64 = (0..nq).map(|q| w[q] * p[q]).sum();
    // volume gradient per degree of freedom: δθ_q = Σ gq[c] δu_c (only components i < nv)
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
    // M_q(ai, bj) = −∂Nₐ/∂x_j ∂N_b/∂x_i (i, j < nv: the volumetric components)
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
                        // Σ_q W p̂ M̄ (M̄ shared) − Σ_q W_q p̂_q M_q
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
