extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::type_id::*;
use crate::term_pointer::*;
use noisy_float::prelude::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum TermReference {
    FuncRef(TermPointer),
    VecRef(Array1<R32>)
}
