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
use crate::newly_evaluated_terms::*;

///Trait which gives a "signature" for
///functions to be included in a [`crate::PrimitiveDirectory`].
///This consists of a name, a collection of required argument types,
///and a return type. Primitive functions are assumed to be 
///uniquely identifiable from their particular implementation of this trait.
pub trait HasFuncSignature {
    ///Gets the name for the implemented function.
    fn get_name(&self) -> String;
    ///Gets the return type for the implemented function.
    fn ret_type(&self) -> TypeId;
    ///Gets the list of required argument types for the implemented function.
    fn required_arg_types(&self) -> Vec::<TypeId>;

    ///Given a collection of [`TermReference`]s, determines whether
    ///we have sufficiently-many arguments to fully evaluate this function.
    fn ready_to_evaluate(&self, args : &Vec::<TermReference>) -> bool {
        let expected_num : usize =  self.required_arg_types().len();
        expected_num == args.len()
    }

    ///Given a [`TypeInfoDirectory`], obtains the [`TypeId`] which
    ///corresponds to the type of this implemented function.
    fn func_type(&self, type_info_directory : &TypeInfoDirectory) -> TypeId {
        let mut reverse_arg_types : Vec<TypeId> = self.required_arg_types();
        reverse_arg_types.reverse();

        let mut result : TypeId = self.ret_type();
        for arg_type_id in reverse_arg_types.drain(..) {
            result = type_info_directory.get_func_type_id(arg_type_id, result);
        }
        result
    }
}

///Trait for primitive function implementations.
pub trait FuncImpl : HasFuncSignature {
    ///Given a handle on the current [`InterpreterState`] (primarily useful if additional terms
    ///need to be evaluated / looked up) and the collection of [`TermReference`] arguments to
    ///apply this function implementation to, yields a [`TermReference`] to the result, along
    ///with a collection of any `NewlyEvaluatedTerms` which may have arisen as part of the
    ///implementation of this method. See `func_impl.rs` in the source for sample implementations.
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>)
                -> (TermReference, NewlyEvaluatedTerms);
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

///Trait to ease implementation of primitive binary operators which have identical argument types
///and return type. To be used in tandem with [`BinaryFuncImpl`].
pub trait BinaryArrayOperator {
    ///Given two arrays of equal dimension, act to yield an array of the same number of dimensions.
    fn act(&self, arg_one : ArrayView1::<R32>, arg_two : ArrayView1::<R32>) -> Array1::<R32>;
    ///Gets the name of this binary operator
    fn get_name(&self) -> String;
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

///[`BinaryArrayOperator`] for vector addition.
pub struct AddOperator {
}

impl BinaryArrayOperator for AddOperator {
    fn act(&self, arg_one : ArrayView1::<R32>, arg_two : ArrayView1::<R32>) -> Array1::<R32> {
        &arg_one + &arg_two
    }
    fn get_name(&self) -> String {
        String::from("+")
    }
}

///[`BinaryArrayOperator`] for vector subtraction.
pub struct SubOperator {
}

impl BinaryArrayOperator for SubOperator {
    fn act(&self, arg_one : ArrayView1::<R32>, arg_two : ArrayView1::<R32>) -> Array1::<R32> {
        &arg_one - &arg_two
    }
    fn get_name(&self) -> String {
        String::from("-")
    }
}

///[`BinaryArrayOperator`] for elementwise vector multiplication.
pub struct MulOperator {
}

impl BinaryArrayOperator for MulOperator {
    fn act(&self, arg_one : ArrayView1::<R32>, arg_two : ArrayView1::<R32>) -> Array1::<R32> {
        &arg_one * &arg_two
    }
    fn get_name(&self) -> String {
        String::from("*")
    }
}

///Wrapper around a [`BinaryArrayOperator`] to conveniently lift it to a [`FuncImpl`]
///given the [`TypeId`] of the argument/return type.
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec::<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::VecRef(_, arg_one_vec) = &args[0] {
            if let TermReference::VecRef(_, arg_two_vec) = &args[1] {
                let result_vec = self.f.act(arg_one_vec.view(), arg_two_vec.view());
                let result_ref = TermReference::VecRef(self.elem_type, result_vec);
                (result_ref, NewlyEvaluatedTerms::new())
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

///Implementation of a "rotate left one index" [`FuncImpl`] for a given vector [`TypeId`].
///(That is, given `[x_1, x_2, ...]`, rotates to `[x_2, x_2, ... x_1]`.
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::VecRef(vector_type, arg_vec) = &args[0] {
            let n = arg_vec.len();
            let arg_vec_head : R32 = arg_vec[[0,]];
            let mut result_vec : Array1::<R32> = Array::from_elem((n,), arg_vec_head);
            for i in 1..n {
                result_vec[[i-1,]] = arg_vec[[i,]];
            }
            let result : TermReference = TermReference::VecRef(*vector_type, result_vec);
            (result, NewlyEvaluatedTerms::new())
        } else {
            panic!();
        }
    }
}

///Implementation of a "set the first element of a vector to the given one" [`FuncImpl`] for the given
///vector and scalar types. 
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::VecRef(vector_type, arg_vec) = &args[0] {
            if let TermReference::VecRef(_, val_vec) = &args[1] {
                let val : R32 = val_vec[[0,]];
                let mut result_vec : Array1<R32> = arg_vec.clone();
                result_vec[[0,]] = val;
                let result = TermReference::VecRef(*vector_type, result_vec);
                (result, NewlyEvaluatedTerms::new())
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

///Implementation of a "get the first element of a vector" [`FuncImpl`] for the given vector
///and scalar types.
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::VecRef(_, arg_vec) = &args[0] {
            let ret_val : R32 = arg_vec[[0,]];
            let result_array : Array1::<R32> = Array::from_elem((1,), ret_val);

            let result = TermReference::VecRef(self.scalar_type, result_array);
            (result, NewlyEvaluatedTerms::new())
        } else {
            panic!();
        }
    }
}

///Implementation of a "function composition" [`FuncImpl`]
#[derive(Clone)]
pub struct ComposeImpl {
    in_type : TypeId,
    middle_type : TypeId,
    func_one : TypeId,
    func_two : TypeId,
    ret_type : TypeId
}

impl ComposeImpl {
    ///Given a [`TypeInfoDirectory`], the input type, a middle type, and a return type,
    ///yields a [`ComposeImpl`] of type `(in -> middle) -> (middle -> return) -> (in -> return)`.
    pub fn new(type_info_directory : &TypeInfoDirectory, 
               in_type : TypeId, middle_type : TypeId, ret_type : TypeId) -> ComposeImpl {
        let func_one : TypeId = type_info_directory.get_func_type_id(middle_type, ret_type);
        let func_two : TypeId = type_info_directory.get_func_type_id(in_type, middle_type);
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
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::FuncRef(func_one) = &args[0] {
            if let TermReference::FuncRef(func_two) = &args[1] {
                let arg : TermReference = args[2].clone();
                let application_one = TermApplication {
                    func_ptr : func_two.clone(),
                    arg_ref : arg
                };
                let (middle_ref, mut newly_evaluated_terms) = state.evaluate(&application_one);
                let application_two = TermApplication {
                    func_ptr : func_one.clone(),
                    arg_ref : middle_ref
                };
                let (final_ref, more_evaluated_terms) = state.evaluate(&application_two);
                newly_evaluated_terms.merge(more_evaluated_terms);
                (final_ref, newly_evaluated_terms)
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

///Implementation of a "Fill a vector with the given scalar" [`FuncImpl`] for the given
///scalar and vector [`TypeId`]s.
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
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        let dim = state.get_context().get_dimension(self.vector_type);

        if let TermReference::VecRef(_, arg_vec) = &args[0] {
            let arg_val : R32 = arg_vec[[0,]];
            let ret_val : Array1::<R32> = Array::from_elem((dim,), arg_val);

            let result = TermReference::VecRef(self.vector_type, ret_val);
            (result, NewlyEvaluatedTerms::new())
        } else {
            panic!();
        }
    }
}

///Implementation of the constant function for the given "return" and "ignored" types.
///The result is of type `return -> ignored -> return`.
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
    fn evaluate(&self, _state : &mut InterpreterState, args : Vec::<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        let result_ptr : TermReference = args[0].clone();
        (result_ptr, NewlyEvaluatedTerms::new())
    }
}

///Implementation of a "reduce this vector by this binary operator to yield a scalar"
///[`FuncImpl`] for the given [`TypeId`]s of the binary scalar operator, the scalar type,
///and the vector type.
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
    fn evaluate(&self, state : &mut InterpreterState, args : Vec<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        let dim = state.get_context().get_dimension(self.vector_type);

        let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();
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
                    let (curry_ref, more_evaluated_terms) = state.evaluate(&term_app_one);
                    newly_evaluated_terms.merge(more_evaluated_terms);

                    if let TermReference::FuncRef(curry_ptr) = curry_ref {
                        let term_app_two = TermApplication {
                            func_ptr : curry_ptr,
                            arg_ref : accum_ref
                        };
                        let (result_ref, more_evaluated_terms) = state.evaluate(&term_app_two);
                        newly_evaluated_terms.merge(more_evaluated_terms);
                        accum_ref = result_ref;
                    } else {
                        panic!();
                    }
                }

                (accum_ref, newly_evaluated_terms)
            } else {
                panic!();
            }
        } else {
            panic!();
        }
    }
}

///Implementation of a "map this scalar function over every element of a vector"
///[`FuncImpl`] for the given scalar type, scalar function type, and vector type.
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
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermReference>) -> (TermReference, NewlyEvaluatedTerms) {
        if let TermReference::FuncRef(func_ptr) = &args[0] {
            if let TermReference::VecRef(_, arg_vec) = &args[1] {
                let n = arg_vec.len();
                let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();
                let mut result : Array1<R32> = Array::from_elem((n,), R32::new(0.0)); 
                for i in 0..n {
                    let boxed_scalar : Array1<R32> = Array::from_elem((1,), arg_vec[i]);
                    let arg_ref = TermReference::VecRef(self.scalar_type, boxed_scalar);

                    let term_app = TermApplication {
                        func_ptr : func_ptr.clone(),
                        arg_ref : arg_ref
                    };
                    let (result_ref, more_evaluated_terms) = state.evaluate(&term_app);
                    newly_evaluated_terms.merge(more_evaluated_terms);
                    if let TermReference::VecRef(_, result_scalar_vec) = result_ref {
                        result[[i,]] = result_scalar_vec[[0,]];
                    }
                }
                let result_ref = TermReference::VecRef(self.vector_type, result);
                (result_ref, newly_evaluated_terms)
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
    use crate::test_utils::*;
    use crate::array_utils::*;

    fn term_ref(in_array : Array1<f32>) -> TermReference {
        let noisy_array = to_noisy(in_array.view());
        if (in_array.shape()[0] == 1) {
            TermReference::VecRef(TEST_SCALAR_T, noisy_array)
        } else {
            TermReference::VecRef(TEST_VECTOR_T, noisy_array)
        }
    }

    #[test]
    fn test_addition() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![3.0f32, 4.0f32])];

        let addition_func = BinaryFuncImpl {
            elem_type : TEST_VECTOR_T,
            f : Box::new(AddOperator {})
        };

        let (result, _) = addition_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![4.0f32, 6.0f32].view());
    }
    #[test]
    fn test_rotate() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);
        let args = vec![term_ref(array![5.0f32, 10.0f32])];

        let rotate_func = RotateImpl {
            vector_type : TEST_VECTOR_T
        };

        let (result, _) = rotate_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![10.0f32, 5.0f32].view());
    }

    #[test]
    fn test_set_head() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);
        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![9.0f32])];

        let set_head_func = SetHeadImpl {
            vector_type : TEST_VECTOR_T,
            scalar_type : TEST_SCALAR_T
        };

        let (result, _) = set_head_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![9.0f32, 2.0f32].view());
    }

    #[test]
    fn test_head() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);

        let args = vec![term_ref(array![1.0f32, 2.0f32])];
        
        let head_func = HeadImpl {
            vector_type : TEST_VECTOR_T,
            scalar_type : TEST_SCALAR_T
        };
        
        let (result, _) = head_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![1.0f32].view());
    }

    #[test]
    fn test_fill() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);
        let args = vec![term_ref(array![3.0f32])];

        let fill_func = FillImpl {
            vector_type : TEST_VECTOR_T,
            scalar_type : TEST_SCALAR_T
        };

        let (result, _) = fill_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![3.0f32, 3.0f32].view());
    }
    
    #[test]
    fn test_const() {
        let ctxt = get_test_vector_only_context();
        let mut state = InterpreterState::new(&ctxt);

        let args = vec![term_ref(array![1.0f32, 2.0f32]), term_ref(array![3.0f32])];

        let const_func = ConstImpl {
            ret_type : TEST_VECTOR_T,
            ignored_type : TEST_SCALAR_T
        };

        let (result, _) = const_func.evaluate(&mut state, args);
        assert_equal_vector_term(result, array![1.0f32, 2.0f32].view());
    }

}
