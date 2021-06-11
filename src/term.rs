use crate::func_impl::*;
use crate::term_reference::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::displayable_with_state::*;
use crate::displayable_with_context::*;
use crate::interpreter_state::*;
use crate::primitive_term_pointer::*;

use serde::{Serialize, Deserialize};

///The application of some [`PrimitiveTermPointer`]
///(which points to a [`FuncImpl`]) to some number of
///[`TermReference`] arguments.
#[derive(Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartiallyAppliedTerm {
    pub func_ptr : PrimitiveTermPointer,
    pub args : Vec<TermReference> 
}

impl DisplayableWithState for PartiallyAppliedTerm {
    fn display(&self, state : &InterpreterState) -> String {
        let mut result = String::from(""); 
        let func_formatted : String = self.func_ptr.display(state.get_context());
        result.push_str(&func_formatted);
        result.push_str("("); 

        let mut done_once : bool = false;

        for arg in self.args.iter() {
            if (done_once) {
                result.push_str(", ");
            }
            let arg_formatted : String = arg.display(state);
            result.push_str(&arg_formatted);
            done_once = true;
        }
        result.push_str(")");
        result
    }
}

impl PartiallyAppliedTerm {
    ///Constructs a new [`PartiallyAppliedTerm`] with the given `func_ptr`
    ///as its function, and an initially-empty list of [`TermReference`] arguments.
    pub fn new(func_ptr : PrimitiveTermPointer) -> PartiallyAppliedTerm {
        PartiallyAppliedTerm {
            func_ptr,
            args : Vec::new()
        }
    }
}
