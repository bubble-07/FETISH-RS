use std::hash::*;
use crate::term_application::*;
use crate::term_pointer::*;
use crate::type_id::*;

#[derive(PartialEq, Hash, Eq, Debug)]
pub struct TermApplicationResult {
    pub term_app : TermApplication,
    pub result_ptr : TermPointer
}

impl TermApplicationResult {
    pub fn get_arg_type(&self) -> TypeId {
        self.term_app.get_arg_type()
    }
    pub fn get_func_type(&self) -> TypeId {
        self.term_app.get_func_type()
    }
    pub fn get_ret_type(&self) -> TypeId {
        self.result_ptr.type_id
    }
    pub fn get_arg_ptr(&self) -> TermPointer {
        self.term_app.arg_ptr.clone()
    }
    pub fn get_ret_ptr(&self) -> TermPointer {
        self.result_ptr.clone()
    }
    pub fn get_func_ptr(&self) -> TermPointer {
        self.term_app.func_ptr.clone()
    }

}
