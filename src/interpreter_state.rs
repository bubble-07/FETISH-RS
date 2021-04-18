extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use std::collections::HashMap;
use crate::nonprimitive_term_pointer::*;
use crate::newly_evaluated_terms::*;
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
    pub ctxt : &'a Context
}

impl <'a> InterpreterState<'a> {
    pub fn get_context(&self) -> &Context {
        self.ctxt
    }

    pub fn store_term(&mut self, type_id : TypeId, term : PartiallyAppliedTerm) -> NonPrimitiveTermPointer {
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&type_id).unwrap();
        let result = type_space.add(term);
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

    pub fn evaluate_application_chain(&mut self, app_chain : &ApplicationChain) -> (TermReference, NewlyEvaluatedTerms) {
        let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();
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

            let eval_pair = self.evaluate(&term_app);
            current_ref = eval_pair.0;
            newly_evaluated_terms.merge(eval_pair.1);
        }
        (current_ref, newly_evaluated_terms)
    }

    pub fn evaluate(&mut self, term_app : &TermApplication) -> (TermReference, NewlyEvaluatedTerms) {
        let func_type_id : TypeId = term_app.get_func_type();

        let func_term : PartiallyAppliedTerm = self.get(term_app.func_ptr);
        let arg_ref : TermReference = term_app.arg_ref.clone();

        let func_impl = self.ctxt.get_primitive(func_term.func_ptr);
        let mut args_copy = func_term.args.clone();

        args_copy.push(arg_ref);

        let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();

        let result_ref : TermReference = if (func_impl.ready_to_evaluate(&args_copy)) {
            let (ret_ref, more_evaluated_terms) = func_impl.evaluate(self, args_copy);
            newly_evaluated_terms.merge(more_evaluated_terms);
            ret_ref
        } else {
            let result = PartiallyAppliedTerm {
                func_ptr : func_term.func_ptr.clone(),
                args : args_copy
            };
            let ret_type_id : TypeId = term_app.get_ret_type(self.ctxt);
            let ret_ptr = self.store_term(ret_type_id, result);

            newly_evaluated_terms.add_term(ret_ptr);

            let ret_ref = TermReference::FuncRef(TermPointer::from(ret_ptr));
            ret_ref
        };
        let application_table : &mut ApplicationTable = self.application_tables.get_mut(&func_type_id).unwrap();

        let term_app_result = TermApplicationResult {
            term_app : term_app.clone(),
            result_ref : result_ref.clone()
        };

        newly_evaluated_terms.add_term_app_result(term_app_result);

        application_table.link(term_app.clone(), result_ref.clone());
        (result_ref, newly_evaluated_terms)
    }

    pub fn ensure_every_type_has_a_term_on_init(&mut self) -> NewlyEvaluatedTerms {
	let mut type_to_term = HashMap::<TypeId, TermReference>::new();
        //Initial population
        for i in 0..self.ctxt.get_total_num_types() {
            let type_id = i as TypeId;
            let kind = self.ctxt.get_type(type_id);
            match (kind) {
                Type::VecType(n) => {
                    type_to_term.insert(type_id, TermReference::VecRef(type_id, Array::zeros((n,))));
                },
                Type::FuncType(_, _) => {
                    let primitive_space = self.ctxt.primitive_directory.primitive_type_spaces.get(&type_id).unwrap();
                    
                    if (primitive_space.terms.len() > 0) {
                        let func_ptr = TermPointer {
                            type_id : type_id,
                            index : TermIndex::Primitive(0)
                        };
                        type_to_term.insert(type_id, TermReference::FuncRef(func_ptr));
                    }
                }
            }
        }
        let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();
        loop {
            let mut found_something = false;
            for i in 0..self.ctxt.get_total_num_types() {
                let func_type_id = i as TypeId;

                if let Option::Some(func_term) = type_to_term.get(&func_type_id) {
                    if let Type::FuncType(arg_type_id, ret_type_id) = self.ctxt.get_type(func_type_id) {
                        if let Option::Some(arg_ref) = type_to_term.get(&arg_type_id) {
                            if (!type_to_term.contains_key(&ret_type_id)) {
                                if let TermReference::FuncRef(func_ptr) = func_term {
                                    let application = TermApplication {
                                        func_ptr : func_ptr.clone(),
                                        arg_ref : arg_ref.clone()
                                    };
                                    let (result_ref, more_evaluated_terms) = self.evaluate(&application);
                                    newly_evaluated_terms.merge(more_evaluated_terms);
                                    type_to_term.insert(ret_type_id, result_ref);

                                    found_something = true;
                                }
                            }
                        }
                    }
                }
            }
            if (!found_something) {
                break;
            }
        }
        newly_evaluated_terms
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
            ctxt
        };

        result
    }
}
