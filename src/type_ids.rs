use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[derive(PartialEq, Hash, Eq, Debug)]
pub struct FuncType {
    pub arg_type : Box<TypeId>,
    pub ret_type : Box<TypeId>
}

#[derive(PartialEq, Hash, Eq, Debug)]
pub enum TypeId {
    FuncId(FuncType),
    VecId(usize)  
}


impl TypeId {
    pub fn can_apply_to(&self, other : &TypeId) -> bool {
        match (self) {
            TypeId::FuncId(func) => &*func.arg_type == other,
            TypeId::VecId(x) => false
        }
    }
}

