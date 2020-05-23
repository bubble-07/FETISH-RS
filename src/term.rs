use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use crate::func_impl::*;
use crate::term_pointer::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use noisy_float::prelude::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum Term {
    VectorTerm(Array1::<R32>),
    PartiallyAppliedTerm(EnumFuncImpl, Vec<TermPointer>)
}
