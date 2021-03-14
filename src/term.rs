use crate::func_impl::*;
use crate::term_reference::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct PartiallyAppliedTerm {
    pub func_impl : EnumFuncImpl,
    pub args : Vec<TermReference> 
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
    pub fn new(func_impl : EnumFuncImpl) -> PartiallyAppliedTerm {
        PartiallyAppliedTerm {
            func_impl,
            args : Vec::new()
        }
    }
}

