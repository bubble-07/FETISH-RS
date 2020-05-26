use crate::term_pointer::*;
use crate::type_ids::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[derive(PartialEq, Hash, Eq, Debug)]
pub struct TermApplication {
    pub func_ptr : TermPointer,
    pub arg_ptr : TermPointer     
}

impl TermApplication {
    pub fn get_func_type(&self) -> &TypeId {
        &self.func_ptr.type_id
    }

    pub fn get_ret_type(&self) -> &TypeId {
        let func_id : &TypeId = self.get_func_type();
        if let TypeId::FuncId(func_type) = func_id {
            &*func_type.ret_type
        } else {
            panic!();
        }
    }
    pub fn get_index_pair(&self) -> (usize, usize) {
        (self.func_ptr.index, self.arg_ptr.index)
    }
}
