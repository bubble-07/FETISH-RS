extern crate ndarray;
extern crate ndarray_linalg;

use crate::type_ids::*;
use crate::interpreter_state::*;
use crate::term_pointer::*;
use crate::term::*;
use crate::term_application::*;
use enum_dispatch::*;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use noisy_float::prelude::*;

use std::rc::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[enum_dispatch]
pub trait HasFuncSignature : Clone + PartialEq + Hash + Eq + Debug {
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;

    fn ready_to_evaluate(&self, args : &Vec::<TermPointer>) -> bool {
        let expected_num : usize =  self.required_arg_types().len();
        expected_num == args.len()
    }
}

#[enum_dispatch]
pub trait FuncImpl : HasFuncSignature {
    fn evaluate(&self, state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, TermPointer);
}

trait FuncImplYieldingTerm : HasFuncSignature {
    fn evaluate_yield_term(&self, state : InterpreterState, 
                                  args : Vec::<TermPointer>) -> (InterpreterState, Term);
}

impl<T : FuncImplYieldingTerm> FuncImpl for T {
    fn evaluate(&self, mut state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, TermPointer) {
        let (state_mod_one, term) = self.evaluate_yield_term(state, args);
        let ret_type : TypeId = self.ret_type();
        state_mod_one.store_term(&ret_type, term)
    }
}

#[enum_dispatch(FuncImpl)]
#[enum_dispatch(HasFuncSignature)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum EnumFuncImpl {    
    MapImpl,
    ConstImpl,
    ComposeImpl,
    FillImpl,
    SetHeadImpl,
    HeadImpl,
    RotateImpl
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct RotateImpl {
    n : usize
}

impl HasFuncSignature for RotateImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![TypeId::VecId(self.n)]
    }
    fn ret_type(&self) -> TypeId {
        TypeId::VecId(self.n)
    }
}

impl FuncImplYieldingTerm for RotateImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            let arg_vec_head : R32 = arg_vec[[0,]];
            let mut result_vec : Array1::<R32> = Array::from_elem((self.n,), arg_vec_head);
            for i in 1..self.n {
                result_vec[[i-1,]] = arg_vec[[i,]];
            }
            let result : Term = Term::VectorTerm(result_vec);
            (state, result)
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct SetHeadImpl {
    n : usize
}

impl HasFuncSignature for SetHeadImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![TypeId::VecId(self.n), TypeId::VecId(1)]
    }
    fn ret_type(&self) -> TypeId {
        TypeId::VecId(1)
    }
}
impl FuncImplYieldingTerm for SetHeadImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        let val_term : &Term = state.get(&args[1]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            if let Term::VectorTerm(val_vec) = val_term {
                let val : R32 = val_vec[[0,]];
                let mut result_vec : Array1<R32> = arg_vec.clone();
                result_vec[[0,]] = val;
                let result : Term = Term::VectorTerm(result_vec);
                (state, result)
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct HeadImpl {
    n : usize
}

impl HasFuncSignature for HeadImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![TypeId::VecId(self.n)]
    }
    fn ret_type(&self) -> TypeId {
        TypeId::VecId(1)
    }
}
impl FuncImplYieldingTerm for HeadImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            let ret_val : R32 = arg_vec[[0,]];
            let result_array : Array1::<R32> = Array::from_elem((1,), ret_val);
            let result : Term = Term::VectorTerm(result_array);

            (state, result)
        } else {
            panic!();
        }
    }
}


#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct ComposeImpl {
    in_type : TypeId,
    middle_type : TypeId,
    ret_type : TypeId
}

impl HasFuncSignature for ComposeImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        let func_one : TypeId = getFuncId(self.middle_type.clone(), self.ret_type.clone()); 
        let func_two : TypeId = getFuncId(self.in_type.clone(), self.middle_type.clone());
        vec![func_one, func_two, self.in_type.clone()]
    }
    fn ret_type(&self) -> TypeId {
        self.ret_type.clone()
    }
}

impl FuncImpl for ComposeImpl {
    fn evaluate(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, TermPointer) {
        let func_one : TermPointer = args[0].clone();
        let func_two : TermPointer = args[1].clone();
        let arg : TermPointer = args[2].clone();
        let application_one = TermApplication {
            func_ptr : func_two,
            arg_ptr : arg
        };
        let (state_mod_one, middle_ptr) = state.evaluate(&application_one);
        let application_two = TermApplication {
            func_ptr : func_one,
            arg_ptr : middle_ptr
        };
        state_mod_one.evaluate(&application_two)
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct FillImpl {
    n : usize
}

impl HasFuncSignature for FillImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![TypeId::VecId(1)]
    }
    fn ret_type(&self) -> TypeId {
        TypeId::VecId(self.n)
    }
}
impl FuncImplYieldingTerm for FillImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            let arg_val : R32 = arg_vec[[0,]];
            let ret_val : Array1::<R32> = Array::from_elem((self.n,), arg_val);
            let ret_term : Term = Term::VectorTerm(ret_val);
            (state, ret_term)
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct ConstImpl {
    ret_type : TypeId,
    ignored_type : TypeId
}

impl HasFuncSignature for ConstImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.ret_type.clone(), self.ignored_type.clone()]
    }
    fn ret_type(&self) -> TypeId {
        self.ret_type.clone()
    }
}
impl FuncImpl for ConstImpl {
    fn evaluate(&self, mut state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, TermPointer) {
        let result_ptr : TermPointer = args[1].clone();
        (state, result_ptr)
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct MapImpl {
    n : usize
}

impl HasFuncSignature for MapImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        let vec_one = Rc::new(TypeId::VecId(1));
        let vec_n = TypeId::VecId(self.n);
        let unary_fn = getFuncIdFromRcs(&vec_one, &vec_one);

        vec![unary_fn, vec_n]
    }
    fn ret_type(&self) -> TypeId {
        TypeId::VecId(self.n)
    }
}
impl FuncImplYieldingTerm for MapImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, Term) {
        let arg_vec_term : Term = state.get(&args[1]).clone();
        let unary_vec_type = TypeId::VecId(1);
        if let Term::VectorTerm(arg_vec) = arg_vec_term {
            let mut result : Array1<R32> = Array::from_elem((self.n,), R32::new(0.0)); 
            for i in 0..self.n {
                let boxed_scalar : Array1<R32> = Array::from_elem((1,), arg_vec[i]);
                let arg_term = Term::VectorTerm(boxed_scalar);
                let (state_mod_one, arg_ptr) = state.store_term(&unary_vec_type, arg_term);
                let term_app = TermApplication {
                    func_ptr : args[0].clone(),
                    arg_ptr 
                };
                let (state_mod_two, result_ptr) = state_mod_one.evaluate(&term_app);
                if let Term::VectorTerm(result_scalar_vec) = state_mod_two.get(&result_ptr) {
                    result[[i,]] = result_scalar_vec[[0,]];
                }
                state = state_mod_two;
            }
            let result_term = Term::VectorTerm(result);
            (state, result_term)
        } else {
            panic!();
        }

    }
}

