extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::array_utils::*;
use crate::type_id::*;
use crate::term_pointer::*;
use crate::displayable_with_state::*;
use crate::interpreter_state::*;
use noisy_float::prelude::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum TermReference {
    FuncRef(TermPointer),
    VecRef(Array1<R32>)
}

impl From<&Array1<f32>> for TermReference {
    fn from(vec : &Array1<f32>) -> Self {
        TermReference::VecRef(to_noisy(vec))
    }
}

impl DisplayableWithState for TermReference {
    fn display(&self, state : &InterpreterState) -> String {
        match (self) {
            TermReference::FuncRef(ptr) => ptr.display(state),
            TermReference::VecRef(vec) => vec.to_string()
        }
    }
}
