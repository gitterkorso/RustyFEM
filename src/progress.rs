//! Framsteg och avbrott för en körning.
//!
//! Den som kör `steps::run` (t.ex. appens session) kan sätta en krok för den
//! egna tråden med `with_hook`. Steg-ramverket rapporterar steg, inkrement och
//! iterationer; kroken svarar `false` för att avbryta, och körningen avslutas
//! då med ett fel mellan två iterationer. Utan krok gör rapporterna ingenting,
//! så resultaten påverkas inte.

use crate::model::{fail, Result};
use std::cell::RefCell;

/// En händelse under körningen.
#[derive(Debug, Clone)]
pub enum Event<'a> {
    /// Ett steg börjar (0-baserat index, namn, procedur).
    Step { index: usize, name: &'a str, procedure: &'a str },
    /// Ett inkrement är godkänt: stegtidsandel 0–1 och antal iterationer.
    Increment { fraction: f64, iterations: usize },
    /// En Newton-iteration: nummer i inkrementet och största residual.
    Iteration { iteration: usize, residual: f64 },
}

type Hook = Box<dyn FnMut(&Event) -> bool>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Kör `f` med `hook` som mottagare av händelser på den här tråden.
pub fn with_hook<T>(hook: impl FnMut(&Event) -> bool + 'static, f: impl FnOnce() -> T) -> T {
    let old = HOOK.with(|h| h.borrow_mut().replace(Box::new(hook)));
    let out = f();
    HOOK.with(|h| *h.borrow_mut() = old);
    out
}

/// Rapporterar en händelse. Fel om kroken begär avbrott.
pub fn emit(e: Event) -> Result<()> {
    let go_on = HOOK.with(|h| match h.borrow_mut().as_mut() {
        Some(f) => f(&e),
        None => true,
    });
    if go_on { Ok(()) } else { fail(CANCELLED) }
}

/// Felmeddelandet för en avbruten körning.
pub const CANCELLED: &str = "körningen avbröts";
