use std::cmp::*;
use std::fmt::*;
use std::hash::*;
use crate::type_id::*;
use crate::term::*;
use crate::term_pointer::*;
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

    pub fn get(&self, term_index : usize) -> &Term {
        &self.terms[term_index]
    }

    ///Adds a given term to this type-space if it doesn't
    ///already exist in that space
    pub fn add(&mut self, term : Term) -> TermPointer {
        if (self.term_to_index_map.contains_key(&term)) {
            let index : usize = *(self.term_to_index_map.get(&term).unwrap());
            TermPointer {
                type_id : self.my_type.clone(),
                index : index
            }
        } else {
            let new_ind : usize = self.terms.len();
            self.terms.push(term.clone());
            self.term_to_index_map.insert(term, new_ind);
            TermPointer {
                type_id : self.my_type.clone(),
                index : new_ind
            }
        }
    }
}
