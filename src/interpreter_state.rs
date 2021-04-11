extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use std::collections::HashMap;
use crate::type_id::*;
use crate::application_chain::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::context::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use topological_sort::TopologicalSort;

pub struct InterpreterState<'a> {
    pub application_tables : HashMap::<TypeId, ApplicationTable<'a>>,
    pub type_spaces : HashMap::<TypeId, TypeSpace>,
    pub new_term_app_results : Vec::<TermApplicationResult>,
    pub new_terms : Vec::<TermPointer>,
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

    pub fn store_term(&mut self, type_id : TypeId, term : PartiallyAppliedTerm) -> TermPointer {
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&type_id).unwrap();
        let result = type_space.add(term);

        self.new_terms.push(result.clone());
        result
    }

    pub fn get(&self, term_ptr : &TermPointer) -> &PartiallyAppliedTerm {
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

    pub fn get_app_results_with_func(&self, func : &TermPointer) -> Vec<TermApplicationResult> {
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

        let application_table : &ApplicationTable = self.application_tables.get(&func_type_id).unwrap();

        if (application_table.has_computed(&term_app)) {
            let result : TermReference = application_table.get_computed(&term_app);
            result
        } else {
            let func_term : PartiallyAppliedTerm = {
                let func_space : &TypeSpace = self.type_spaces.get(&func_type_id).unwrap();
                func_space.get(term_app.func_ptr.index).clone()
            };
            let arg_ref : TermReference = term_app.arg_ref.clone();

            let func_impl = func_term.func_impl;
            let mut args_copy = func_term.args.clone();

            args_copy.push(arg_ref);

            let result_ref : TermReference = if (func_impl.ready_to_evaluate(&args_copy)) {
                func_impl.evaluate(self, args_copy)
            } else {
                let result = PartiallyAppliedTerm {
                    func_impl : func_impl,
                    args : args_copy
                };
                let ret_type_id : TypeId = term_app.get_ret_type(self.ctxt);
                let ret_ptr = self.store_term(ret_type_id, result);
                let ret_ref = TermReference::FuncRef(ret_ptr);
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
    }
    
    pub fn ensure_every_term_has_an_application(&mut self) {
        let mut topo_sort = TopologicalSort::<TypeId>::new();
        for i in 0..self.ctxt.get_total_num_types() {
            let type_id = i as TypeId;
            if let Type::FuncType(arg_type_id, ret_type_id) = self.ctxt.get_type(type_id) {
                topo_sort.add_dependency(type_id, arg_type_id);
                topo_sort.add_dependency(type_id, ret_type_id);
            }
        }

        while (topo_sort.len() > 0) {
            let mut type_ids = topo_sort.pop_all();
            for func_type_id in type_ids.drain(..) {
                if let Type::FuncType(arg_type_id, _) = self.ctxt.get_type(func_type_id) {
                    let func_space = self.type_spaces.get(&func_type_id).unwrap();

                    for index in 0..func_space.get_num_terms() {
                        let func_ptr = TermPointer {
                            type_id : func_type_id,
                            index
                        };
                        let num_existing_app_results = {
                            let app_table = self.application_tables.get(&func_type_id).unwrap();
                            let existing_app_results = app_table.get_app_results_with_func(&func_ptr);  
                            existing_app_results.len()
                        };

                        if (num_existing_app_results == 0) {
                            let arg_ref = match (self.ctxt.get_type(arg_type_id)) {
                                Type::FuncType(_, _) => {
                                    let arg_space = self.type_spaces.get(&arg_type_id).unwrap();
                                    let arg_ptr = arg_space.draw_random_ptr().unwrap();
                                    TermReference::FuncRef(arg_ptr)
                                },
                                Type::VecType(dim) => {
                                    TermReference::VecRef(arg_type_id, Array::zeros((dim,)))
                                }
                            };
                            let term_app = TermApplication {
                                func_ptr : func_ptr.clone(),
                                arg_ref : arg_ref
                            };

                            self.evaluate(&term_app);
                        }
                    }
                }
            }
        }
    }

    fn ensure_every_type_has_a_term(&mut self) {
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
                    let type_space = self.type_spaces.get(&type_id).unwrap();
                    let maybe_func_ptr = type_space.draw_random_ptr();
                    if let Option::Some(func_ptr) = maybe_func_ptr {
                        type_to_term.insert(type_id, TermReference::FuncRef(func_ptr));
                    }
                }
            }
        }
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
                                    let result_ref = self.evaluate(&application);
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
    }

    pub fn add_init(&mut self, func : Box<dyn FuncImpl>) -> TermPointer {
        let func_type_id : TypeId = func.func_type(&self.ctxt.type_info_directory);
        let type_space : &mut TypeSpace = self.type_spaces.get_mut(&func_type_id).unwrap();
        let result = type_space.add_init(func);

        self.new_terms.push(result.clone());

        result
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

        let mut result = InterpreterState {
            application_tables,
            type_spaces,
            new_term_app_results : Vec::new(),
            new_terms : Vec::new(),
            ctxt
        };

        //Now populate the type spaces with the known function implementations
        //using TypeSpace#add(PartiallyAppliedTerm)
        
        //TODO: Add primitives
        for type_id in 0..ctxt.get_total_num_types() {
            if (!ctxt.is_vector_type(type_id)) {
                let primitive_type_space = ctxt.primitive_directory.primitive_type_spaces.get(&type_id).unwrap();
                for term in primitive_type_space.terms.iter() {
                    result.add_init(term.clone());
                }
            }
        }
       
        result.ensure_every_type_has_a_term();
        result.ensure_every_term_has_an_application();

        result
    }
}
