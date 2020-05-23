use crate::term_pointer::*;
use std::cmp::*;
use std::fmt::*;
use std::hash::*;

#[derive(PartialEq, Hash, Eq, Debug)]
pub struct TermApplication {
    func_ptr : TermPointer,
    arg_ptr : TermPointer     
}
