use crate::term_pointer::*;
use crate::context::*;
use crate::term_reference::*;
use crate::type_id::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

#[derive(Clone, PartialEq, Hash, Eq)]
pub struct TermApplication {
    pub func_ptr : TermPointer,
    pub arg_ref : TermReference
}

impl TermApplication {
    pub fn get_arg_type(&self, ctxt : &Context) -> TypeId {
        let (arg_type, _) = self.get_func_type_pair(ctxt);
        arg_type
    }

    pub fn get_ret_type(&self, ctxt : &Context) -> TypeId {
        let (_, ret_type) = self.get_func_type_pair(ctxt);
        ret_type
    }

    pub fn get_func_type(&self) -> TypeId {
        self.func_ptr.type_id
    }

    fn get_func_type_pair(&self, ctxt : &Context) -> (TypeId, TypeId) {
        let func_id : TypeId = self.get_func_type();
        let func_type : Type = ctxt.get_type(func_id);
        if let Type::FuncType(arg_id, ret_id) = func_type {
            (arg_id, ret_id)
        } else {
            panic!();
        }
    }
}

impl DisplayableWithState for TermApplication {
    fn display(&self, state : &InterpreterState) -> String {
        format!("{} {}", self.func_ptr.display(state), self.arg_ref.display(state))
    }
}
