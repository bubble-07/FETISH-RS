use crate::type_id::*;
use crate::interpreter_state::*;
use crate::term_application::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use std::collections::HashMap;

pub struct ApplicationTable {
    func_space : TypeId,
    arg_space : TypeId,
    result_space : TypeId,
    table : HashMap::<TermApplication, TermReference>
}

impl ApplicationTable {
    pub fn new(func_space : TypeId) -> ApplicationTable {
        let func_type = get_type(func_space);
        if let Type::FuncType(arg_space, result_space) = func_type {
            ApplicationTable {
                func_space : func_space,
                arg_space : arg_space,
                result_space : result_space,
                table : HashMap::new()
            }
        } else {
            panic!();
        }
    }

    pub fn has_computed(&self, term_app : &TermApplication) -> bool {
        self.table.contains_key(term_app)
    }

    pub fn get_computed(&self, term_app : &TermApplication) -> TermReference {
        self.table.get(term_app).unwrap().clone()
    }

    pub fn link(&mut self, term_app : TermApplication, result_ref : TermReference) {
        self.table.insert(term_app, result_ref);
    }
}
