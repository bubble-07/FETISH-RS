extern crate ndarray;
extern crate ndarray_linalg;

use crate::type_id::*;
use crate::interpreter_state::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term::*;
use crate::term_application::*;
use enum_dispatch::*;
use crate::test_utils::*;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use noisy_float::prelude::*;

use std::rc::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::params::*;

#[enum_dispatch]
pub trait HasFuncSignature {
    fn get_name(&self) -> String;
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;

    fn ready_to_evaluate(&self, args : &Vec::<TermReference>) -> bool {
        let expected_num : usize =  self.required_arg_types().len();
        expected_num == args.len()
    }

    fn func_type(&self) -> TypeId {
        let mut reverse_arg_types : Vec<TypeId> = self.required_arg_types();
        reverse_arg_types.reverse();

        let mut result : TypeId = self.ret_type();
        for arg_type_id in reverse_arg_types.drain(..) {
            let result_type : Type = Type::FuncType(arg_type_id, result);
            result = get_type_id(&result_type);
        }
        result
    }
}

#[enum_dispatch]
pub trait FuncImpl : HasFuncSignature {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference;
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
    RotateImpl,
    ReduceImpl
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
    pub elem_type : TypeId,
    pub f : EnumBinaryArrayOperator
}

impl HasFuncSignature for BinaryFuncImpl {
    fn get_name(&self) -> String {
        String::from(match (self.f) {
            EnumBinaryArrayOperator::AddOperator(_) => "+",
            EnumBinaryArrayOperator::SubOperator(_) => "-",
            EnumBinaryArrayOperator::MulOperator(_) => "*"
        })
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.elem_type, self.elem_type]
    }
    fn ret_type(&self) -> TypeId {
        self.elem_type
    }
}

impl FuncImpl for BinaryFuncImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        if let TermReference::VecRef(arg_one_vec) = &args[0] {
            if let TermReference::VecRef(arg_two_vec) = &args[1] {
                let result_vec = self.f.act(&arg_one_vec, &arg_two_vec);
                let result_ref = TermReference::VecRef(result_vec);
                result_ref
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
    fn get_name(&self) -> String {
        String::from("rotate")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}

impl FuncImpl for RotateImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(arg_vec) = &args[0] {
            let n = arg_vec.len();
            let arg_vec_head : R32 = arg_vec[[0,]];
            let mut result_vec : Array1::<R32> = Array::from_elem((n,), arg_vec_head);
            for i in 1..n {
                result_vec[[i-1,]] = arg_vec[[i,]];
            }
            let result : TermReference = TermReference::VecRef(result_vec);
            result
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct SetHeadImpl {
}

impl HasFuncSignature for SetHeadImpl {
    fn get_name(&self) -> String {
        String::from("setHead")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T, *SCALAR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}
impl FuncImpl for SetHeadImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(arg_vec) = &args[0] {
            if let TermReference::VecRef(val_vec) = &args[1] {
                let val : R32 = val_vec[[0,]];
                let mut result_vec : Array1<R32> = arg_vec.clone();
                result_vec[[0,]] = val;
                let result = TermReference::VecRef(result_vec);
                result
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
    fn get_name(&self) -> String {
        String::from("head")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *SCALAR_T
    }
}
impl FuncImpl for HeadImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(arg_vec) = &args[0] {
            let ret_val : R32 = arg_vec[[0,]];
            let result_array : Array1::<R32> = Array::from_elem((1,), ret_val);

            let result = TermReference::VecRef(result_array);
            result
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

impl ComposeImpl {
    pub fn new(in_type : TypeId, middle_type : TypeId, ret_type : TypeId) -> ComposeImpl {
        let func_one : TypeId = get_type_id(&Type::FuncType(middle_type, ret_type));
        let func_two : TypeId = get_type_id(&Type::FuncType(in_type, middle_type));
        ComposeImpl {
            in_type,
            middle_type,
            func_one,
            func_two,
            ret_type
        }
    }
}

impl HasFuncSignature for ComposeImpl {
    fn get_name(&self) -> String {
        String::from("compose")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.func_one, self.func_two, self.in_type]
    }
    fn ret_type(&self) -> TypeId {
        self.ret_type
    }
}

impl FuncImpl for ComposeImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::FuncRef(func_one) = &args[0] {
            if let TermReference::FuncRef(func_two) = &args[1] {
                let arg : TermReference = args[2].clone();
                let application_one = TermApplication {
                    func_ptr : func_two.clone(),
                    arg_ref : arg
                };
                let middle_ref = state.evaluate(&application_one);
                let application_two = TermApplication {
                    func_ptr : func_one.clone(),
                    arg_ref : middle_ref
                };
                state.evaluate(&application_two)
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct FillImpl {
}

impl HasFuncSignature for FillImpl {
    fn get_name(&self) -> String {
        String::from("fill")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*SCALAR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}
impl FuncImpl for FillImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(arg_vec) = &args[0] {
            let arg_val : R32 = arg_vec[[0,]];
            let ret_val : Array1::<R32> = Array::from_elem((DIM,), arg_val);

            let result = TermReference::VecRef(ret_val);
            result
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct ConstImpl {
    pub ret_type : TypeId,
    pub ignored_type : TypeId
}

impl HasFuncSignature for ConstImpl {
    fn get_name(&self) -> String {
        String::from("const")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.ret_type.clone(), self.ignored_type.clone()]
    }
    fn ret_type(&self) -> TypeId {
        self.ret_type.clone()
    }
}
impl FuncImpl for ConstImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        let result_ptr : TermReference = args[0].clone();
        result_ptr
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct ReduceImpl {
}

impl HasFuncSignature for ReduceImpl {
    fn get_name(&self) -> String {
        String::from("reduce")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*BINARY_SCALAR_FUNC_T, *SCALAR_T, *VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *SCALAR_T
    }
}

impl FuncImpl for ReduceImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        let mut accum_ref : TermReference = args[1].clone();
        if let TermReference::FuncRef(func_ptr) = &args[0] {
            if let TermReference::VecRef(vec) = &args[2] {
                for i in 0..DIM {
                    //First, put the scalar term at this position into a term ref
                    let val : R32 = vec[[i,]];
                    let val_vec : Array1::<R32> = Array::from_elem((1,), val);
                    let val_ref = TermReference::VecRef(val_vec);
                     
                    let term_app_one = TermApplication {
                        func_ptr : func_ptr.clone(),
                        arg_ref : val_ref
                    };
                    let curry_ref = state.evaluate(&term_app_one);

                    if let TermReference::FuncRef(curry_ptr) = curry_ref {
                        let term_app_two = TermApplication {
                            func_ptr : curry_ptr,
                            arg_ref : accum_ref
                        };
                        let result_ref = state.evaluate(&term_app_two);
                        accum_ref = result_ref;
                    } else {
                        panic!();
                    }
                }

                accum_ref
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct MapImpl {
}

impl HasFuncSignature for MapImpl {
    fn get_name(&self) -> String {
        String::from("map")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![*UNARY_SCALAR_FUNC_T, *VECTOR_T]
    }
    fn ret_type(&self) -> TypeId {
        *VECTOR_T
    }
}

impl FuncImpl for MapImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        let unary_vec_type = *VECTOR_T;
        if let TermReference::FuncRef(func_ptr) = &args[0] {
            if let TermReference::VecRef(arg_vec) = &args[1] {
                let n = arg_vec.len();
                let mut result : Array1<R32> = Array::from_elem((n,), R32::new(0.0)); 
                for i in 0..n {
                    let boxed_scalar : Array1<R32> = Array::from_elem((1,), arg_vec[i]);
                    let arg_ref = TermReference::VecRef(boxed_scalar);

                    let term_app = TermApplication {
                        func_ptr : func_ptr.clone(),
                        arg_ref : arg_ref
                    };
                    let result_ref = state.evaluate(&term_app);
                    if let TermReference::VecRef(result_scalar_vec) = result_ref {
                        result[[i,]] = result_scalar_vec[[0,]];
                    }
                }
                let result_ref = TermReference::VecRef(result);
                result_ref
            } else {
                panic!();
            }
        } else {
            panic!();
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![3.0f32, 4.0f32])];

        let addition_func = BinaryFuncImpl {
            elem_type : *VECTOR_T,
            f : EnumBinaryArrayOperator::AddOperator(AddOperator {})
        };

        let result = addition_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![4.0f32, 6.0f32]);
    }
    #[test]
    fn test_rotate() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![5.0f32, 10.0f32])];

        let rotate_func = RotateImpl {};

        let result = rotate_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![10.0f32, 5.0f32]);
    }

    #[test]
    fn test_set_head() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![9.0f32])];

        let set_head_func = SetHeadImpl {};

        let result = set_head_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![9.0f32, 2.0f32]);
    }

    #[test]
    fn test_head() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![1.0f32, 2.0f32])];
        
        let head_func = HeadImpl {};
        
        let result = head_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![1.0f32]);
    }

    #[test]
    fn test_fill() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![3.0f32])];

        let fill_func = FillImpl {};

        let result = fill_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![3.0f32, 3.0f32]);
    }
    
    #[test]
    fn test_const() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![3.0f32])];

        let const_func = ConstImpl {
            ret_type : *VECTOR_T,
            ignored_type : *SCALAR_T
        };

        let result = const_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![1.0f32, 2.0f32]);
    }

}
