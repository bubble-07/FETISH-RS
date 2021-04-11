extern crate ndarray;
extern crate ndarray_linalg;

use crate::type_id::*;
use crate::interpreter_state::*;
use crate::term_reference::*;
use crate::term_application::*;

use ndarray::*;
use noisy_float::prelude::*;

use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::params::*;

pub trait HasFuncSignature {
    fn get_name(&self) -> String;
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;

    fn ready_to_evaluate(&self, args : &Vec::<TermReference>) -> bool {
        let expected_num : usize =  self.required_arg_types().len();
        expected_num == args.len()
    }

    fn func_type(&self, type_info_directory : &TypeInfoDirectory) -> TypeId {
        let mut reverse_arg_types : Vec<TypeId> = self.required_arg_types();
        reverse_arg_types.reverse();

        let mut result : TypeId = self.ret_type();
        for arg_type_id in reverse_arg_types.drain(..) {
            let result_type : Type = Type::FuncType(arg_type_id, result);
            result = type_info_directory.get(&result_type);
        }
        result
    }
}

pub trait FuncImpl : HasFuncSignature {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference;
    fn clone_box(&self) -> Box<dyn FuncImpl>;
}

impl PartialEq for dyn FuncImpl + '_ {
    fn eq(&self, other : &Self) -> bool {
        self.get_name() == other.get_name() &&
        self.required_arg_types() == other.required_arg_types() &&
        self.ret_type() == other.ret_type()
    }
}

impl Eq for dyn FuncImpl + '_ {}

impl Hash for dyn FuncImpl + '_ {
    fn hash<H : Hasher>(&self, state : &mut H) {
        self.get_name().hash(state);
        self.required_arg_types().hash(state);
        self.ret_type().hash(state);
    }
}

impl Clone for Box<dyn FuncImpl> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub trait BinaryArrayOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32>;
    fn get_name(&self) -> String;
    fn clone_box(&self) -> Box<dyn BinaryArrayOperator>;
}

impl PartialEq for dyn BinaryArrayOperator + '_ {
    fn eq(&self, other : &Self) -> bool {
        self.get_name() == other.get_name()
    }
}

impl Eq for dyn BinaryArrayOperator + '_ {}

impl Hash for dyn BinaryArrayOperator + '_ {
    fn hash<H : Hasher>(&self, state : &mut H) {
        self.get_name().hash(state);
    }
}

impl Clone for Box<dyn BinaryArrayOperator> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub struct AddOperator {
}

impl BinaryArrayOperator for AddOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one + arg_two
    }
    fn get_name(&self) -> String {
        String::from("+")
    }
    fn clone_box(&self) -> Box<dyn BinaryArrayOperator> {
        Box::new(AddOperator {})
    }
}

pub struct SubOperator {
}

impl BinaryArrayOperator for SubOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one - arg_two
    }
    fn get_name(&self) -> String {
        String::from("-")
    }
    fn clone_box(&self) -> Box<dyn BinaryArrayOperator> {
        Box::new(SubOperator {})
    }
}

pub struct MulOperator {
}

impl BinaryArrayOperator for MulOperator {
    fn act(&self, arg_one : &Array1::<R32>, arg_two : &Array1::<R32>) -> Array1::<R32> {
        arg_one * arg_two
    }
    fn get_name(&self) -> String {
        String::from("*")
    }
    fn clone_box(&self) -> Box<dyn BinaryArrayOperator> {
        Box::new(MulOperator {})
    }
}

#[derive(Clone)]
pub struct BinaryFuncImpl {
    pub elem_type : TypeId,
    pub f : Box<dyn BinaryArrayOperator>
}

impl HasFuncSignature for BinaryFuncImpl {
    fn get_name(&self) -> String {
        self.f.get_name()
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.elem_type, self.elem_type]
    }
    fn ret_type(&self) -> TypeId {
        self.elem_type
    }
}

impl FuncImpl for BinaryFuncImpl {
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        if let TermReference::VecRef(_, arg_one_vec) = &args[0] {
            if let TermReference::VecRef(_, arg_two_vec) = &args[1] {
                let result_vec = self.f.act(&arg_one_vec, &arg_two_vec);
                let result_ref = TermReference::VecRef(self.elem_type, result_vec);
                result_ref
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct RotateImpl {
    pub vector_type : TypeId
}

impl HasFuncSignature for RotateImpl {
    fn get_name(&self) -> String {
        String::from("rotate")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.vector_type]
    }
    fn ret_type(&self) -> TypeId {
        self.vector_type
    }
}

impl FuncImpl for RotateImpl {
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(vector_type, arg_vec) = &args[0] {
            let n = arg_vec.len();
            let arg_vec_head : R32 = arg_vec[[0,]];
            let mut result_vec : Array1::<R32> = Array::from_elem((n,), arg_vec_head);
            for i in 1..n {
                result_vec[[i-1,]] = arg_vec[[i,]];
            }
            let result : TermReference = TermReference::VecRef(*vector_type, result_vec);
            result
        } else {
            panic!();
        }
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct SetHeadImpl {
    pub vector_type : TypeId,
    pub scalar_type : TypeId
}

impl HasFuncSignature for SetHeadImpl {
    fn get_name(&self) -> String {
        String::from("setHead")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.vector_type, self.scalar_type]
    }
    fn ret_type(&self) -> TypeId {
        self.vector_type
    }
}
impl FuncImpl for SetHeadImpl {
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(vector_type, arg_vec) = &args[0] {
            if let TermReference::VecRef(_, val_vec) = &args[1] {
                let val : R32 = val_vec[[0,]];
                let mut result_vec : Array1<R32> = arg_vec.clone();
                result_vec[[0,]] = val;
                let result = TermReference::VecRef(*vector_type, result_vec);
                result
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct HeadImpl {
    pub vector_type : TypeId,
    pub scalar_type : TypeId
}

impl HasFuncSignature for HeadImpl {
    fn get_name(&self) -> String {
        String::from("head")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.vector_type]
    }
    fn ret_type(&self) -> TypeId {
        self.scalar_type
    }
}
impl FuncImpl for HeadImpl {
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        if let TermReference::VecRef(_, arg_vec) = &args[0] {
            let ret_val : R32 = arg_vec[[0,]];
            let result_array : Array1::<R32> = Array::from_elem((1,), ret_val);

            let result = TermReference::VecRef(self.scalar_type, result_array);
            result
        } else {
            panic!();
        }
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ComposeImpl {
    in_type : TypeId,
    middle_type : TypeId,
    func_one : TypeId,
    func_two : TypeId,
    ret_type : TypeId
}

impl ComposeImpl {
    pub fn new(type_info_directory : &TypeInfoDirectory, 
               in_type : TypeId, middle_type : TypeId, ret_type : TypeId) -> ComposeImpl {
        let func_one : TypeId = type_info_directory.get(&Type::FuncType(middle_type, ret_type));
        let func_two : TypeId = type_info_directory.get(&Type::FuncType(in_type, middle_type));
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
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct FillImpl {
    pub scalar_type : TypeId,
    pub vector_type : TypeId
}

impl HasFuncSignature for FillImpl {
    fn get_name(&self) -> String {
        String::from("fill")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.scalar_type]
    }
    fn ret_type(&self) -> TypeId {
        self.vector_type
    }
}
impl FuncImpl for FillImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        let dim = state.get_context().get_dimension(self.vector_type);

        if let TermReference::VecRef(_, arg_vec) = &args[0] {
            let arg_val : R32 = arg_vec[[0,]];
            let ret_val : Array1::<R32> = Array::from_elem((dim,), arg_val);

            let result = TermReference::VecRef(self.vector_type, ret_val);
            result
        } else {
            panic!();
        }
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        let result_ptr : TermReference = args[0].clone();
        result_ptr
    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct ReduceImpl {
    pub binary_scalar_func_type : TypeId,
    pub scalar_type : TypeId,
    pub vector_type : TypeId
}

impl HasFuncSignature for ReduceImpl {
    fn get_name(&self) -> String {
        String::from("reduce")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.binary_scalar_func_type, self.scalar_type, self.vector_type]
    }
    fn ret_type(&self) -> TypeId {
        self.scalar_type
    }
}

impl FuncImpl for ReduceImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> TermReference {
        let dim = state.get_context().get_dimension(self.vector_type);

        let mut accum_ref : TermReference = args[1].clone();
        if let TermReference::FuncRef(func_ptr) = &args[0] {
            if let TermReference::VecRef(_, vec) = &args[2] {
                for i in 0..dim {
                    //First, put the scalar term at this position into a term ref
                    let val : R32 = vec[[i,]];
                    let val_vec : Array1::<R32> = Array::from_elem((1,), val);
                    let val_ref = TermReference::VecRef(self.scalar_type, val_vec);
                     
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
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct MapImpl {
    pub scalar_type : TypeId,
    pub unary_scalar_func_type : TypeId,
    pub vector_type : TypeId
}

impl HasFuncSignature for MapImpl {
    fn get_name(&self) -> String {
        String::from("map")
    }
    fn required_arg_types(&self) -> Vec<TypeId> {
        vec![self.unary_scalar_func_type, self.vector_type]
    }
    fn ret_type(&self) -> TypeId {
        self.vector_type
    }
}

impl FuncImpl for MapImpl {
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> TermReference {
        if let TermReference::FuncRef(func_ptr) = &args[0] {
            if let TermReference::VecRef(_, arg_vec) = &args[1] {
                let n = arg_vec.len();
                let mut result : Array1<R32> = Array::from_elem((n,), R32::new(0.0)); 
                for i in 0..n {
                    let boxed_scalar : Array1<R32> = Array::from_elem((1,), arg_vec[i]);
                    let arg_ref = TermReference::VecRef(self.scalar_type, boxed_scalar);

                    let term_app = TermApplication {
                        func_ptr : func_ptr.clone(),
                        arg_ref : arg_ref
                    };
                    let result_ref = state.evaluate(&term_app);
                    if let TermReference::VecRef(_, result_scalar_vec) = result_ref {
                        result[[i,]] = result_scalar_vec[[0,]];
                    }
                }
                let result_ref = TermReference::VecRef(self.vector_type, result);
                result_ref
            } else {
                panic!();
            }
        } else {
            panic!();
        }

    }
    fn clone_box(&self) -> Box<dyn FuncImpl> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_addition() {
        let mut state = InterpreterState::new();
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![3.0f32, 4.0f32])];

        let addition_func = BinaryFuncImpl {
            elem_type : *VECTOR_T,
            f : Box::new(AddOperator {})
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
