extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::array_utils::*;
use crate::type_id::*;
use crate::term_pointer::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;
use noisy_float::prelude::*;

///A reference to an arbitrary term, which may belong to
///a function [`Type`] or a vector [`Type`].
///Vectors are stored inline here, whereas functions
///are stored as [`TermPointer`]s to the relevant
///[`PartiallyAppliedTerm`]s in an [`InterpreterState`].
#[derive(Clone, PartialEq, Hash, Eq)]
pub enum TermReference {
    ///A [`TermPointer`] reference to a function
    FuncRef(TermPointer),
    ///A vector of the given [`TypeId`] with the given elements.
    VecRef(TypeId, Array1<R32>)
}

impl TermReference {
    ///Gets the [`TypeId`] of the term for this [`TermReference`].
    pub fn get_type(&self) -> TypeId {
        match (&self) {
            TermReference::FuncRef(func_ptr) => func_ptr.type_id,
            TermReference::VecRef(type_id, _) => *type_id
        }
    }
}

impl DisplayableWithState for TermReference {
    fn display(&self, state : &InterpreterState) -> String {
        match (self) {
            TermReference::FuncRef(ptr) => ptr.display(state),
            TermReference::VecRef(_, vec) => vec.to_string()
        }
    }
}
