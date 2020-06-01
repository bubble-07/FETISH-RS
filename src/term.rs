use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use crate::func_impl::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use noisy_float::prelude::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct PartiallyAppliedTerm {
    pub func_impl : EnumFuncImpl,
    pub args : Vec<TermReference> 
}
