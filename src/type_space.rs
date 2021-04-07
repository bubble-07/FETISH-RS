use crate::type_id::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::func_impl::*;
use rand::Rng;
use std::collections::HashMap;

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

    pub fn draw_random_ptr(&self) -> Option<TermPointer> {
        if (self.terms.len() == 0) {
            Option::None
        } else {
            let mut rng = rand::thread_rng();
            let index : usize = rng.gen_range(0, self.terms.len());
            let result = TermPointer {
                type_id : self.my_type.clone(),
                index : index
            };
            Option::Some(result)
        }
    }
    
    pub fn get(&self, term_index : usize) -> &PartiallyAppliedTerm {
        &self.terms[term_index]
    }

    pub fn add_init(&mut self, func : Box<dyn FuncImpl>) -> TermPointer {
        let term = PartiallyAppliedTerm::new(func);
        self.add(term)
    }

    ///Adds a given term to this type-space if it doesn't
    ///already exist in that space
    pub fn add(&mut self, term : PartiallyAppliedTerm) -> TermPointer {
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
                type_id : self.my_type,
                index : new_ind
            }
        }
    }
}
