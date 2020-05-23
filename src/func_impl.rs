use crate::type_ids::*;
use crate::interpreter_state::*;
use crate::term_pointer::*;
use crate::term::*;
use enum_dispatch::*;

use std::cmp::*;
use std::fmt::*;
use std::hash::*;

pub trait FuncImpl : PartialEq + Hash + Eq + Debug {
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermPointer>) -> Term;
}

#[enum_dispatch(FuncImpl)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum EnumFuncImpl {
    
}
