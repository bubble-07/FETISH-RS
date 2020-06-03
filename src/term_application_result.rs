use std::hash::*;
use crate::term_application::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TermApplicationResult {
    pub term_app : TermApplication,
    pub result_ref : TermReference
}

impl TermApplicationResult {
    pub fn get_arg_type(&self) -> TypeId {
        self.term_app.get_arg_type()
    }
    pub fn get_func_type(&self) -> TypeId {
        self.term_app.get_func_type()
    }
    pub fn get_ret_type(&self) -> TypeId {
        self.term_app.get_ret_type()
    }
    pub fn get_arg_ref(&self) -> TermReference {
        self.term_app.arg_ref.clone()
    }
    pub fn get_ret_ref(&self) -> TermReference {
        self.result_ref.clone()
    }
    pub fn get_func_ptr(&self) -> TermPointer {
        self.term_app.func_ptr.clone()
    }

}
