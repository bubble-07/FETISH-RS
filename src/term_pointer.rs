use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_ids::*;

#[derive(PartialEq, Hash, Eq, Debug)]
pub struct TermPointer {
    type_id : TypeId,
    index : usize
}
