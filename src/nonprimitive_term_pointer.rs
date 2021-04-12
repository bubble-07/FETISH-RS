use crate::term_pointer::*;
use crate::type_id::*;
use crate::displayable_with_state::*;
use crate::context::*;
use crate::interpreter_state::*;

#[derive(Copy, Clone, PartialEq, Hash, Eq)]
pub struct NonPrimitiveTermPointer {
    pub type_id : TypeId,
    pub index : usize
}

impl DisplayableWithState for NonPrimitiveTermPointer {
    fn display(&self, state : &InterpreterState) -> String {
        let term = state.get_nonprimitive(*self);
        term.display(state)
    }
}