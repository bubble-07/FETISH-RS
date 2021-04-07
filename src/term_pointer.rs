use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_id::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TermPointer {
    pub type_id : TypeId,
    pub index : usize
}

impl DisplayableWithState for TermPointer {
    fn display(&self, state : &InterpreterState) -> String {
        let term = state.get(self);
        term.display(state)
    }
}
