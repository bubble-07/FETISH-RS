use crate::type_ids::*;
use crate::interpreter_state::*;
use crate::term_pointer::*;
use crate::term::*;
use enum_dispatch::*;

use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[enum_dispatch]
pub trait FuncImpl : Clone + PartialEq + Hash + Eq + Debug {
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;
    fn evaluate(&self, state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, Term);

    fn ready_to_evaluate(&self, args : &Vec::<TermPointer>) -> bool {
        let expected_num : usize =  self.required_arg_types().len();
        expected_num == args.len()
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct IdImpl {
    ret_type : TypeId
}

impl FuncImpl for IdImpl {
    fn ret_type(&self) -> TypeId {
        self.ret_type.clone()
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        let mut result : Vec::<TypeId> = Vec::new();
        result.push(self.ret_type.clone());
        result
    }
    fn evaluate(&self, state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, Term) {
        let result : Term = state.get(&args[0]).clone();
        (state, result)
    }
}

#[enum_dispatch(FuncImpl)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum EnumFuncImpl {    
    IdImpl
}
