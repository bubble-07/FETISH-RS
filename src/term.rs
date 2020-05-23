use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use crate::func_impl::*;
use crate::term_pointer::*;

pub enum Term {
    VectorTerm(Array1<f32>),
    PartiallyAppliedTerm(Box<FuncImpl>, Vec<TermPointer>)
}
