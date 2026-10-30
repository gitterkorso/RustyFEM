//! Gränssnitt för linjära ekvationslösare.
//!
//! Kärnan beskriver systemets struktur som en graf över "block" (noder med
//! fria frihetsgrader). Lösaren väljer ekvationsordning (`first_dof`), tar
//! emot den undre triangeln av en symmetrisk matris och löser K x = b.
//! Samma matrisobjekt återanvänds mellan Newton-iterationer: `zero`,
//! assemblering med `add`, `factor`, `solve`.

use crate::model::Result;

/// Systemets struktur: ett block per nod med fria frihetsgrader.
pub struct Graph {
    /// Blockens koordinater (för geometrisk ordning).
    pub coords: Vec<[f64; 3]>,
    /// Grannblock (utan blocket självt), sorterade.
    pub adj: Vec<Vec<usize>>,
    /// Antal ekvationer i varje block.
    pub ndof: Vec<usize>,
}

pub trait LinearSolver: Send + Sync {
    /// Namn för val och utskrift, t.ex. "direct".
    fn name(&self) -> &'static str;
    /// Symbolisk analys: ger en nollställd matris att assemblera i.
    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>>;
}

pub trait SystemMatrix: Send {
    /// Antal ekvationer.
    fn n(&self) -> usize;
    /// Ekvationsnummer för första frihetsgraden i varje block; blockets övriga
    /// frihetsgrader följer direkt efter.
    fn first_dof(&self) -> &[usize];
    fn zero(&mut self);
    /// Adderar `v` till (r, c) i undre triangeln (r ≥ c).
    fn add(&mut self, r: usize, c: usize, v: f64);
    /// Faktoriserar. `Err(ekvation)` om matrisen är singulär eller inte
    /// positivt definit vid den ekvationen.
    fn factor(&mut self) -> std::result::Result<(), usize>;
    /// Löser K x = b på plats (efter `factor`).
    fn solve(&self, b: &mut [f64]);

    /// Adderar `v` till (r, c) i en osymmetrisk matris (r och c godtyckliga).
    /// `add(r, c, v)` betyder alltid ett symmetriskt bidrag (även (c, r));
    /// en symmetrisk lösare behåller bara den undre triangeln.
    fn add_unsym(&mut self, r: usize, c: usize, v: f64) {
        if r >= c {
            self.add(r, c, v);
        }
    }
    /// `false` om lösaren lagrar och löser en osymmetrisk matris.
    fn is_symmetric(&self) -> bool {
        true
    }
    /// Tröghet (antal negativa, noll- och positiva pivoter) efter `factor`,
    /// om lösaren beräknar den. Ger varningen om negativa egenvärden.
    fn inertia(&self) -> Option<Inertia> {
        None
    }
}

/// Tröghet hos en faktoriserad symmetrisk matris (Sylvesters tröghetslag).
pub use crate::sparse::indef::Inertia;
