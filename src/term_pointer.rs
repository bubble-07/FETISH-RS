use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_id::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;
use crate::term_index::*;
use crate::primitive_term_pointer::*;
use crate::nonprimitive_term_pointer::*;

#[derive(Clone, PartialEq, Hash, Eq)]
pub struct TermPointer {
    pub type_id : TypeId,
    pub index : TermIndex
}

impl DisplayableWithState for TermPointer {
    fn display(&self, state : &InterpreterState) -> String {
        let term = state.get(self);
        term.display(state)
    }
}

impl From<NonPrimitiveTermPointer> for TermPointer {
    fn from(term_ptr : NonPrimitiveTermPointer) -> Self {
        TermPointer {
            type_id : term_ptr.type_id,
            index : TermIndex::NonPrimitive(term_ptr.index)
        }
    }
}
impl From<PrimitiveTermPointer> for TermPointer {
    fn from(term_ptr : PrimitiveTermPointer) -> Self {
        TermPointer {
            type_id : term_ptr.type_id,
            index : TermIndex::Primitive(term_ptr.index)
        }
    }
}
