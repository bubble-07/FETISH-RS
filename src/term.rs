use crate::func_impl::*;
use crate::term_reference::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;

#[derive(Clone, Hash, Eq)]
pub struct PartiallyAppliedTerm {
    pub func_impl : Box<dyn FuncImpl>,
    pub args : Vec<TermReference> 
}

impl PartialEq for PartiallyAppliedTerm {
    fn eq(&self, other : &Self) -> bool {
        &self.func_impl == &other.func_impl &&
        &self.args == &other.args
    }
}

impl DisplayableWithState for PartiallyAppliedTerm {
    fn display(&self, state : &InterpreterState) -> String {
        let mut result = String::from(""); 
        let func_formatted : String = self.func_impl.get_name();
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
    pub fn new(func_impl : Box<dyn FuncImpl>) -> PartiallyAppliedTerm {
        PartiallyAppliedTerm {
            func_impl,
            args : Vec::new()
        }
    }
}
