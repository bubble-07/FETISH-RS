use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use std::rc::*;

#[derive(Eq, PartialEq, Hash, Debug)]
pub struct FuncType {
    pub arg_type : Rc::<TypeId>,
    pub ret_type : Rc::<TypeId>
}

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub enum TypeId {
    FuncId(FuncType),
    VecId(usize)  
}

impl Clone for FuncType {
    fn clone(&self) -> Self {
        FuncType {
            arg_type : Rc::clone(&self.arg_type),
            ret_type : Rc::clone(&self.ret_type)
        }
    }
}


impl TypeId {
    pub fn can_apply_to(&self, other : &TypeId) -> bool {
        match (self) {
            TypeId::FuncId(func) => &*func.arg_type == other,
            TypeId::VecId(x) => false
        }
    }
}

