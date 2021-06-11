use crate::term_reference::*;

use serde::{Serialize, Deserialize};

#[derive(Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct TermInputOutput {
    pub input : TermReference,
    pub output : TermReference
}
