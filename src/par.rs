//! Enkla parallella hjälpfunktioner ovanpå std::thread::scope.

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// `f(0..n)` parallellt med bevarad ordning.
pub fn map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, threads: usize, f: F) -> Vec<T> {
    let t = threads.min(n / 16).max(1);
    if t == 1 {
        return (0..n).map(f).collect();
    }
    let chunk = n.div_ceil(t);
    // användarrutinernas körkontext (steg, inkrement, fält) följer med till arbetstrådarna
    let run = crate::user::context::current();
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = (0..t)
            .map(|i| {
                let run = run.clone();
                s.spawn(move || {
                    crate::user::context::install(run);
                    (i * chunk..((i + 1) * chunk).min(n)).map(f).collect::<Vec<T>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Kör `f(rad, radbuffert)` för `nrows` rader i `buf` (radlängd `stride`).
pub fn rows_mut<C, F>(buf: &mut [f64], stride: usize, nrows: usize, threads: usize, cost: C, f: F)
where
    C: Fn(usize) -> f64,
    F: Fn(usize, &mut [f64]) + Sync,
{
    blocks_mut(buf, stride, nrows, threads, cost, |r0, chunk| {
        for (i, row) in chunk.chunks_mut(stride).enumerate() {
            f(r0 + i, row);
        }
    });
}

/// Delar `nrows` rader i `buf` (radlängd `stride`) i sammanhängande block med
/// ungefär lika stor kostnad och kör `f(första rad, block)` för varje block.
pub fn blocks_mut<C, F>(buf: &mut [f64], stride: usize, nrows: usize, threads: usize, cost: C, f: F)
where
    C: Fn(usize) -> f64,
    F: Fn(usize, &mut [f64]) + Sync,
{
    let t = threads.min(nrows).max(1);
    if t == 1 {
        f(0, &mut buf[..nrows * stride]);
        return;
    }
    let total: f64 = (0..nrows).map(&cost).sum();
    let mut bounds = vec![0];
    let mut acc = 0.0;
    for r in 0..nrows {
        acc += cost(r);
        if acc >= total * bounds.len() as f64 / t as f64 && bounds.len() < t {
            bounds.push(r + 1);
        }
    }
    bounds.push(nrows);
    bounds.dedup();
    std::thread::scope(|s| {
        let f = &f;
        let mut rest = &mut buf[..nrows * stride];
        for w in bounds.windows(2) {
            let (chunk, tail) = rest.split_at_mut((w[1] - w[0]) * stride);
            rest = tail;
            let r0 = w[0];
            s.spawn(move || f(r0, chunk));
        }
    });
}
