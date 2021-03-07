use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use std::cmp::*;
use std::fmt::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

#[derive(Clone)]
pub struct ApplicationChain {
    pub term_refs : Vec<TermReference>
}
