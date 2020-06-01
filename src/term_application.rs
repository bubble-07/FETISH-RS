use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TermApplication {
    pub func_ptr : TermPointer,
    pub arg_ref : TermReference
}

impl TermApplication {
    pub fn get_arg_type(&self) -> TypeId {
        let (arg_type, _) = self.get_func_type_pair();
        arg_type
    }

    pub fn get_ret_type(&self) -> TypeId {
        let (_, ret_type) = self.get_func_type_pair();
        ret_type
    }

    pub fn get_func_type(&self) -> TypeId {
        self.func_ptr.type_id
    }

    fn get_func_type_pair(&self) -> (TypeId, TypeId) {
        let func_id : TypeId = self.get_func_type();
        let func_type : Type = get_type(func_id);
        if let Type::FuncType(arg_id, ret_id) = func_type {
            (arg_id, ret_id)
        } else {
            panic!();
        }
    }
}
