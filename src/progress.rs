//! Progress and cancellation for a run.
//!
//! Whoever runs `steps::run` (e.g. the app session) can set a hook for its
//! own thread with `with_hook`. The step framework reports steps, increments and
//! iterations; the hook returns `false` to cancel, and the run then ends
//! with an error between two iterations. Without a hook the reports do nothing,
//! so the results are not affected.

use crate::model::{fail, Result};
use std::cell::RefCell;

/// An event during the run.
#[derive(Debug, Clone)]
pub enum Event<'a> {
    /// A step starts (0-based index, name, procedure).
    Step { index: usize, name: &'a str, procedure: &'a str },
    /// An increment is accepted: step time fraction 0–1 and number of iterations.
    Increment { fraction: f64, iterations: usize },
    /// A Newton iteration: number within the increment and largest residual.
    Iteration { iteration: usize, residual: f64 },
}

type Hook = Box<dyn FnMut(&Event) -> bool>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Runs `f` with `hook` receiving the events on this thread.
pub fn with_hook<T>(hook: impl FnMut(&Event) -> bool + 'static, f: impl FnOnce() -> T) -> T {
    let old = HOOK.with(|h| h.borrow_mut().replace(Box::new(hook)));
    let out = f();
    HOOK.with(|h| *h.borrow_mut() = old);
    out
}

/// Reports an event. Error if the hook requests cancellation.
pub fn emit(e: Event) -> Result<()> {
    let go_on = HOOK.with(|h| match h.borrow_mut().as_mut() {
        Some(f) => f(&e),
        None => true,
    });
    if go_on { Ok(()) } else { fail(CANCELLED) }
}

/// The error message for a cancelled run.
pub const CANCELLED: &str = "the run was cancelled";
