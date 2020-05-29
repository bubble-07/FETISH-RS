use std::collections::HashMap;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_application::*;
use crate::func_impl::*;

pub struct InterpreterState {
    application_tables : HashMap::<TypeId, ApplicationTable>,
    type_spaces : HashMap::<TypeId, TypeSpace>
}

impl InterpreterState {

    pub fn store_term(mut self, type_id : TypeId, term : Term) -> (InterpreterState, TermPointer) {
        let mut type_space : &mut TypeSpace = self.type_spaces.get_mut(&type_id).unwrap();
        let result = type_space.add(term);
        (self, result)
    }

    pub fn get(&self, term_ptr : &TermPointer) -> &Term {
        self.type_spaces.get(&term_ptr.type_id).unwrap().get(term_ptr.index)
    }

    pub fn evaluate(mut self, term_app : &TermApplication) -> (InterpreterState, TermPointer) {
        let func_type_id : TypeId = term_app.get_func_type();

        let application_table : &ApplicationTable = self.application_tables.get(&func_type_id).unwrap();

        if (application_table.has_computed(&term_app)) {
            let result : TermPointer = application_table.get_computed(&term_app);
            (self, result)
        } else {
            let func_term : Term = {
                let func_space : &TypeSpace = self.type_spaces.get(&func_type_id).unwrap();
                func_space.get(term_app.func_ptr.index).clone()
            };
            let arg_ptr : TermPointer = term_app.arg_ptr.clone();
            if let Term::PartiallyAppliedTerm(func_impl, args) = func_term {
                let mut args_copy : Vec<TermPointer> = args.clone();
                args_copy.push(arg_ptr);

                let result_tuple : (InterpreterState, TermPointer) = if (func_impl.ready_to_evaluate(&args_copy)) {
                    func_impl.evaluate(self, args_copy)
                } else {
                    let result : Term = Term::PartiallyAppliedTerm(func_impl, args_copy);
                    let ret_type_id : TypeId = term_app.get_ret_type();
                    self.store_term(ret_type_id, result)
                };
                let (mut zelf, result_ptr) = result_tuple;

                let mut application_table : &mut ApplicationTable = zelf.application_tables.get_mut(&func_type_id).unwrap();

                application_table.link(term_app, &result_ptr);

                (zelf, result_ptr)
            } else {
                panic!();
            }
        }

        
    }
}
