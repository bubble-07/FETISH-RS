use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_ids::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TermPointer {
    pub type_id : TypeId,
    pub index : usize
}
