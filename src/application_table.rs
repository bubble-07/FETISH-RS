use crate::type_id::*;
use crate::interpreter_state::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use std::collections::HashMap;
use multimap::MultiMap;

pub struct ApplicationTable {
    func_space : TypeId,
    arg_space : TypeId,
    result_space : TypeId,
    table : HashMap::<TermApplication, TermReference>,
    result_to_application_map :  MultiMap::<TermReference, TermApplicationResult>,
    arg_to_application_map : MultiMap::<TermReference, TermApplicationResult>,
    func_to_application_map : MultiMap::<TermPointer, TermApplicationResult>
}

impl ApplicationTable {
    pub fn new(func_space : TypeId) -> ApplicationTable {
        let func_type = get_type(func_space);
        if let Type::FuncType(arg_space, result_space) = func_type {
            ApplicationTable {
                func_space : func_space,
                arg_space : arg_space,
                result_space : result_space,
                table : HashMap::new(),
                result_to_application_map : MultiMap::new(),
                arg_to_application_map : MultiMap::new(),
                func_to_application_map : MultiMap::new()
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

    pub fn get_app_results_with_arg(&self, arg : &TermReference) -> Vec<TermApplicationResult> {
        match (self.arg_to_application_map.get_vec(arg)) {
            Option::Some(vec) => vec.clone(),
            Option::None => Vec::new()
        }
    }

    pub fn get_app_results_with_func(&self, func : &TermPointer) -> Vec<TermApplicationResult> {
        match (self.func_to_application_map.get_vec(func)) {
            Option::Some(vec) => vec.clone(),
            Option::None => Vec::new()
        }
    }

    pub fn get_app_results_with_result(&self, result : &TermReference) -> Vec<TermApplicationResult> {
        match (self.result_to_application_map.get_vec(result)) {
            Option::Some(vec) => vec.clone(),
            Option::None => Vec::new()
        }
    }

    pub fn link(&mut self, term_app : TermApplication, result_ref : TermReference) {
        let result = TermApplicationResult {
            term_app : term_app.clone(),
            result_ref : result_ref.clone()
        };

        self.result_to_application_map.insert(result_ref.clone(), result.clone());
        self.arg_to_application_map.insert(term_app.arg_ref.clone(), result.clone());
        self.func_to_application_map.insert(term_app.func_ptr.clone(), result.clone());
        self.table.insert(term_app, result_ref);
    }
}
