//! Keywords for contact initialization and controls in general contact
//! (Area contact3). *CONTACT INITIALIZATION DATA and
//! *CONTACT INITIALIZATION ASSIGNMENT are stored in the model and used when the file
//! has been read (`interactions::contact::adjust_slaves`). *CONTACT CONTROLS
//! ASSIGNMENT (STABILIZE, AUTOMATIC TOLERANCES for general contact) is accepted
//! like *CONTACT CONTROLS: they affect the iteration path but not the
//! converged final state.

use crate::inp::{ignore, store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONTACT CONTROLS ASSIGNMENT", kind: Kind::Both, handler: ignore },
    Keyword { name: "CONTACT INITIALIZATION ASSIGNMENT", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT INITIALIZATION DATA", kind: Kind::Model, handler: store_model },
];
