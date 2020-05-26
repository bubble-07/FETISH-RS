use crate::type_ids::*;
use crate::interpreter_state::*;
use crate::term_application::*;
use crate::term_pointer::*;
use std::collections::HashMap;

pub struct ApplicationTable {
    func_space : TypeId,
    arg_space : TypeId,
    result_space : TypeId,
    table : HashMap::<(usize, usize), usize>
}

impl ApplicationTable {
    pub fn has_computed(&self, term_app : &TermApplication) -> bool {
        let term_pair : (usize, usize) = term_app.get_index_pair();
        self.table.contains_key(&term_pair)
    }

    pub fn get_computed(&self, term_app : &TermApplication) -> TermPointer {
        let term_pair : (usize, usize) = term_app.get_index_pair();
        let out_index : usize = *self.table.get(&term_pair).unwrap();
        TermPointer {
            type_id : self.result_space.clone(),
            index : out_index
        }
    }

    pub fn link(&mut self, term_app : &TermApplication, result_ptr : &TermPointer) {
        let term_pair : (usize, usize) = term_app.get_index_pair();
        let result_index : usize = result_ptr.index;
        self.table.insert(term_pair, result_index);
    }
}
