extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use std::collections::HashMap;
use crate::nonprimitive_term_pointer::*;
use crate::type_id::*;
use crate::application_chain::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term_index::*;
use crate::term::*;
use crate::context::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::primitive_term_pointer::*;
use crate::func_impl::*;
use topological_sort::TopologicalSort;

pub struct InterpreterState<'a> {
    pub application_tables : HashMap::<TypeId, ApplicationTable<'a>>,
    pub type_spaces : HashMap::<TypeId, TypeSpace>,
    pub new_term_app_results : Vec::<TermApplicationResult>,
    pub new_terms : Vec::<NonPrimitiveTermPointer>,
    pub ctxt : &'a Context
}

impl <'a> InterpreterState<'a> {
    pub fn get_context(&self) -> &Context {
        self.ctxt
    }

    pub fn clear_newly_received(&mut self) {
        self.new_term_app_results.clear();
        self.new_terms.clear();
    }

    pub fn store_term(&mut self, type_id : TypeId, term : PartiallyAppliedTerm) -> NonPrimitiveTermPointer {
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&type_id).unwrap();
        let result = type_space.add(term);

        self.new_terms.push(result.clone());
        result
    }

    pub fn get(&self, term_ptr : TermPointer) -> PartiallyAppliedTerm {
        match (term_ptr.index) {
            TermIndex::Primitive(index) => {
                let primitive_ptr = PrimitiveTermPointer {
                    type_id : term_ptr.type_id,
                    index : index
                };
                let result = PartiallyAppliedTerm::new(primitive_ptr);
                result
            },
            TermIndex::NonPrimitive(index) => {
                self.type_spaces.get(&term_ptr.type_id).unwrap().get(index).clone()
            }
        }
    }

    pub fn get_nonprimitive(&self, term_ptr : NonPrimitiveTermPointer) -> &PartiallyAppliedTerm {
        self.type_spaces.get(&term_ptr.type_id).unwrap().get(term_ptr.index)
    }

    pub fn get_app_results_with_arg(&self, arg : &TermReference) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_arg(arg);
            result.append(&mut temp);
        }
        result
    }

    pub fn get_app_results_with_func(&self, func : TermPointer) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_func(func);
            result.append(&mut temp);
        }
        result
    }

    pub fn get_app_results_with_result(&self, result_term : &TermReference) -> Vec<TermApplicationResult> {
        let mut result : Vec<TermApplicationResult> = Vec::new();
        for table in self.application_tables.values() {
            let mut temp = table.get_app_results_with_result(result_term);
            result.append(&mut temp);
        }
        result
    }

    pub fn evaluate_application_chain(&mut self, app_chain : &ApplicationChain) -> TermReference {
        let mut current_ref = app_chain.term_refs[0].clone();
        for i in 1..app_chain.term_refs.len() {
            let current_type = current_ref.get_type();

            let other_ref = &app_chain.term_refs[i];
            let other_type = other_ref.get_type();

            let mut current_is_applicative = false;
            if (!self.ctxt.is_vector_type(current_type)) {
                let arg_type = self.ctxt.get_arg_type_id(current_type);
                current_is_applicative = (arg_type == other_type);
            }

            let term_app = if (current_is_applicative) {
                match (current_ref) {
                    TermReference::FuncRef(current_ptr) => {
                        TermApplication {
                            func_ptr : current_ptr,
                            arg_ref : other_ref.clone()
                        }
                    },
                    TermReference::VecRef(_, _) => { panic!(); }
                }
            } else {
                match (other_ref) {
                    TermReference::FuncRef(other_ptr) => {
                        TermApplication {
                            func_ptr : other_ptr.clone(),
                            arg_ref : current_ref
                        }
                    },
                    TermReference::VecRef(_, _) => { panic!(); }
                }
            };

            current_ref = self.evaluate(&term_app);
        }
        current_ref
    }

    pub fn evaluate(&mut self, term_app : &TermApplication) -> TermReference {
        let func_type_id : TypeId = term_app.get_func_type();

        let func_term : PartiallyAppliedTerm = self.get(term_app.func_ptr);
        let arg_ref : TermReference = term_app.arg_ref.clone();

        let func_impl = self.ctxt.get_primitive(func_term.func_ptr);
        let mut args_copy = func_term.args.clone();

        args_copy.push(arg_ref);

        let result_ref : TermReference = if (func_impl.ready_to_evaluate(&args_copy)) {
            func_impl.evaluate(self, args_copy)
        } else {
            let result = PartiallyAppliedTerm {
                func_ptr : func_term.func_ptr.clone(),
                args : args_copy
            };
            let ret_type_id : TypeId = term_app.get_ret_type(self.ctxt);
            let ret_ptr = self.store_term(ret_type_id, result);
            let ret_ref = TermReference::FuncRef(TermPointer::from(ret_ptr));
            ret_ref
        };
        let application_table : &mut ApplicationTable = self.application_tables.get_mut(&func_type_id).unwrap();

        let term_app_result = TermApplicationResult {
            term_app : term_app.clone(),
            result_ref : result_ref.clone()
        };
        self.new_term_app_results.push(term_app_result);

        application_table.link(term_app.clone(), result_ref.clone());
        result_ref
    }

    pub fn new(ctxt : &'a Context) -> InterpreterState<'a> {
        //Initialize hashmaps for each type in the global type table 
        let mut application_tables = HashMap::<TypeId, ApplicationTable>::new();
        let mut type_spaces = HashMap::<TypeId, TypeSpace>::new();

        for type_id in 0..ctxt.get_total_num_types() {
            if (!ctxt.is_vector_type(type_id)) {
                application_tables.insert(type_id, ApplicationTable::new(type_id, ctxt));
                type_spaces.insert(type_id, TypeSpace::new(type_id));
            }
        }

        let result = InterpreterState {
            application_tables,
            type_spaces,
            new_term_app_results : Vec::new(),
            new_terms : Vec::new(),
            ctxt
        };

        result
    }
}
