use crate::type_id::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::term_pointer::*;
use crate::context::*;
use crate::term_reference::*;
use std::collections::HashMap;
use multimap::MultiMap;

///For a given function type `A -> B`, stores
///current information about [`TermApplicationResult`]s
///for that function type with several easily-queryable views.
pub struct ApplicationTable<'a> {
    func_type_id : TypeId,
    table : MultiMap::<TermApplication, TermReference>,
    result_to_application_map :  MultiMap::<TermReference, TermApplicationResult>,
    arg_to_application_map : MultiMap::<TermReference, TermApplicationResult>,
    func_to_application_map : MultiMap::<TermPointer, TermApplicationResult>,
    ctxt : &'a Context
}

impl <'a> ApplicationTable<'a> {
    ///Constructs an initially-empty [`ApplicationTable`] for the given
    ///function [`TypeId`] in the given [`Context`].
    pub fn new(func_type_id : TypeId, ctxt : &'a Context) -> ApplicationTable<'a> {
        if (!ctxt.is_vector_type(func_type_id)) {
            ApplicationTable {
                func_type_id,
                table : MultiMap::new(),
                result_to_application_map : MultiMap::new(),
                arg_to_application_map : MultiMap::new(),
                func_to_application_map : MultiMap::new(),
                ctxt
            }
        } else {
            panic!();
        }
    }

    pub fn get_results_from_application(&self, term_app : &TermApplication) -> Vec<TermReference> {
        match (self.table.get_vec(term_app)) {
            Option::Some(vec) => vec.clone(),
            Option::None => Vec::new()
        }
    }

    pub fn get_app_results_with_arg(&self, arg : &TermReference) -> Vec<TermApplicationResult> {
        match (self.arg_to_application_map.get_vec(arg)) {
            Option::Some(vec) => vec.clone(),
            Option::None => Vec::new()
        }
    }

    pub fn get_app_results_with_func(&self, func : TermPointer) -> Vec<TermApplicationResult> {
        match (self.func_to_application_map.get_vec(&func)) {
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
