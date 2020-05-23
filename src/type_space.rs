use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_ids::*;
use crate::term::*;
use std::collections::HashMap;

pub struct TypeSpace {
    my_type : TypeId,
    terms : Vec<Term>,
    term_to_index_map : HashMap::<Term, usize>
}

impl TypeSpace {
    fn new(id : TypeId) -> TypeSpace {
        TypeSpace {
            my_type : id,
            terms : Vec::<Term>::new(),
            term_to_index_map : HashMap::new()
        }
    }
}
