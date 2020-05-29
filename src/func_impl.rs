extern crate ndarray;
extern crate ndarray_linalg;

use crate::type_id::*;
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
pub trait HasFuncSignature {
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
        state_mod_one.store_term(ret_type, term)
    }
}

#[enum_dispatch(FuncImpl)]
#[enum_dispatch(HasFuncSignature)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum EnumFuncImpl {    
    BinaryFuncImpl, 
    MapImpl,
    ConstImpl,
    ComposeImpl,
    FillImpl,
    SetHeadImpl,
    HeadImpl,
    RotateImpl
}

#[enum_dispatch]
pub trait BinaryArrayOperator : Clone + PartialEq + Hash + Eq + Debug {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32>;
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct AddOperator {
}

impl BinaryArrayOperator for AddOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one + arg_two
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct SubOperator {
}

impl BinaryArrayOperator for SubOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one - arg_two
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct MulOperator {
}

impl BinaryArrayOperator for MulOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one * arg_two
    }
}


#[enum_dispatch(BinaryArrayOperator)]
#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum EnumBinaryArrayOperator {
    AddOperator, 
    SubOperator,
    MulOperator //For now, no "div", because we'd need to deal with nans
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct BinaryFuncImpl {
    elem_type : TypeId,
    f : EnumBinaryArrayOperator
}

impl HasFuncSignature for BinaryFuncImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.elem_type, self.elem_type]
    }
    fn ret_type(&self) -> TypeId {
        self.elem_type
    }
}

impl FuncImplYieldingTerm for BinaryFuncImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, Term) {
        let arg_one_term = state.get(&args[0]);
        let arg_two_term = state.get(&args[1]);
        if let Term::VectorTerm(arg_one_vec) = arg_one_term {
            if let Term::VectorTerm(arg_two_vec) = arg_two_term {
                let result_vec = self.f.act(arg_one_vec, arg_two_vec);
                let result_term = Term::VectorTerm(result_vec); 
                (state, result_term)
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct RotateImpl {
}

impl HasFuncSignature for RotateImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}

impl FuncImplYieldingTerm for RotateImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            let n = arg_vec.len();
            let arg_vec_head : R32 = arg_vec[[0,]];
            let mut result_vec : Array1::<R32> = Array::from_elem((n,), arg_vec_head);
            for i in 1..n {
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
}

impl HasFuncSignature for SetHeadImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T, *SCALAR_T]
    }
    fn ret_type(&self) -> TypeId {
        *SCALAR_T
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
}

impl HasFuncSignature for HeadImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *SCALAR_T
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
    func_one : TypeId,
    func_two : TypeId,
    ret_type : TypeId
}

impl HasFuncSignature for ComposeImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.func_one, self.func_two, self.in_type]
    }
    fn ret_type(&self) -> TypeId {
        self.ret_type
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
}

impl HasFuncSignature for FillImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*SCALAR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}
impl FuncImplYieldingTerm for FillImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec<TermPointer>) -> (InterpreterState, Term) {
        let arg_term : &Term = state.get(&args[0]);
        if let Term::VectorTerm(arg_vec) = arg_term {
            let arg_val : R32 = arg_vec[[0,]];
            let ret_val : Array1::<R32> = Array::from_elem((DIM,), arg_val);
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
}

impl HasFuncSignature for MapImpl {
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*UNARY_SCALAR_FUNC_T, *VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}

impl FuncImplYieldingTerm for MapImpl {
    fn evaluate_yield_term(&self, mut state : InterpreterState, args : Vec::<TermPointer>) -> (InterpreterState, Term) {
        let arg_vec_term : Term = state.get(&args[1]).clone();
        let unary_vec_type = *VECTOR_T;
        if let Term::VectorTerm(arg_vec) = arg_vec_term {
            let n = arg_vec.len();
            let mut result : Array1<R32> = Array::from_elem((n,), R32::new(0.0)); 
            for i in 0..n {
                let boxed_scalar : Array1<R32> = Array::from_elem((1,), arg_vec[i]);
                let arg_term = Term::VectorTerm(boxed_scalar);
                let (state_mod_one, arg_ptr) = state.store_term(unary_vec_type, arg_term);
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

