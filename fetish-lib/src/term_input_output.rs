use crate::term_reference::*;

#[derive(Clone, PartialEq, Hash, Eq)]
pub struct TermInputOutput {
    pub input : TermReference,
    pub output : TermReference
}
