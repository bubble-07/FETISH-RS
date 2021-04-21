use crate::type_id::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::func_impl::*;
use rand::Rng;
use std::collections::HashMap;
use crate::term_index::*;
use crate::nonprimitive_term_pointer::*;

pub struct TypeSpace {
    my_type : TypeId,
    terms : Vec<PartiallyAppliedTerm>,
    term_to_index_map : HashMap::<PartiallyAppliedTerm, usize>
}

impl TypeSpace {
    pub fn new(id : TypeId) -> TypeSpace {
        TypeSpace {
            my_type : id,
            terms : Vec::new(),
            term_to_index_map : HashMap::new()
        }
    }

    pub fn get_num_terms(&self) -> usize {
        self.terms.len()
    }
    
    pub fn get(&self, term_index : usize) -> &PartiallyAppliedTerm {
        &self.terms[term_index]
    }

    ///Adds a given term to this type-space if it doesn't
    ///already exist in that space
    pub fn add(&mut self, term : PartiallyAppliedTerm) -> NonPrimitiveTermPointer {
        if (self.term_to_index_map.contains_key(&term)) {
            let index : usize = *(self.term_to_index_map.get(&term).unwrap());
            NonPrimitiveTermPointer {
                type_id : self.my_type.clone(),
                index
            }
        } else {
            let new_ind : usize = self.terms.len();
            self.terms.push(term.clone());
            self.term_to_index_map.insert(term, new_ind);
            NonPrimitiveTermPointer {
                type_id : self.my_type,
                index : new_ind
            }
        }
    }
}
