extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::rc::*;
use crate::interpreter_state::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use crate::bayes_utils::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::feature_collection::*;
use crate::enum_feature_collection::*;
use topological_sort::TopologicalSort;

pub struct EmbedderState {
    model_spaces : HashMap::<TypeId, ModelSpace>
}

impl EmbedderState {

    pub fn new() -> EmbedderState {
        let mut model_spaces = HashMap::<TypeId, ModelSpace>::new();
        
        let mut dimensions = HashMap::<TypeId, usize>::new();
        let mut topo_sort = TopologicalSort::<TypeId>::new();
        for i in 0..total_num_types() {
            let type_id : TypeId = i as TypeId;
            match get_type(type_id) {
                Type::FuncType(arg_type_id, ret_type_id) => {
                    topo_sort.add_dependency(arg_type_id, type_id);
                    topo_sort.add_dependency(ret_type_id, type_id);
                },
                Type::VecType(dim) => {
                    dimensions.insert(type_id, dim);
                }
            };
        }
        
        while (topo_sort.len() > 0) {
            let mut type_ids : Vec<TypeId> = topo_sort.pop_all();
            for func_type_id in type_ids.drain(..) {
                if let Type::FuncType(arg_type_id, ret_type_id) = get_type(func_type_id) {
                    let arg_dimension = *dimensions.get(&arg_type_id).unwrap();
                    let ret_dimension = *dimensions.get(&ret_type_id).unwrap();

                    let model_space = ModelSpace::new(arg_dimension, ret_dimension);
                    model_spaces.insert(func_type_id, model_space);
                }
            }
        }

        EmbedderState {
            model_spaces
        }
    }

    pub fn get_embedding(&self, term_ptr : &TermPointer) -> &Model {
        let space : &ModelSpace = self.model_spaces.get(&term_ptr.type_id).unwrap();
        space.get_model(term_ptr.index)
    }

    pub fn get_mut_embedding(&mut self, term_ptr : TermPointer) -> &mut Model {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.get_model_mut(term_ptr.index)
    }

    pub fn init_embedding(&mut self, term_ptr : TermPointer) {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.add_model(term_ptr.index)
    }

    fn get_schmear_from_ref(&self, term_ref : &TermReference) -> Schmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_schmear_from_ptr(func_ptr),
            TermReference::VecRef(vec) => Schmear::from_vector(&vec)
        }
    }
    fn get_schmear_from_ptr(&self, term_ptr : &TermPointer) -> Schmear {
        let embedding : &Model = self.get_embedding(term_ptr);
        embedding.get_schmear()
    }

    fn get_inverse_schmear_from_ptr(&self, term_ptr : &TermPointer) -> InverseSchmear {
        let embedding : &Model = self.get_embedding(term_ptr);
        embedding.get_inverse_schmear()
    }

    fn get_inverse_schmear_from_ref(&self, term_ref : &TermReference) -> InverseSchmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_inverse_schmear_from_ptr(func_ptr),
            TermReference::VecRef(vec) => InverseSchmear::ident_precision_from_noisy(vec)
        }
    }

    fn get_mean_from_ref(&self, term_ref : &TermReference) -> Array1<f32> {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_mean_from_ptr(func_ptr),
            TermReference::VecRef(vec) => from_noisy(vec)
        }
    }

    fn get_mean_from_ptr(&self, term_ptr : &TermPointer) -> Array1<f32> {
        let embedding : &Model = self.get_embedding(term_ptr);
        embedding.get_mean_as_vec()
    }

    //Propagagtes prior updates downwards
    fn propagate_prior_recursive(&mut self, interpreter_state : &InterpreterState,
                                 modified : HashSet::<TermPointer>,
                                 all_modified : &mut HashSet::<TermPointer>) {
        let mut follow_up : HashSet::<TermApplicationResult> = HashSet::new();
        for elem in modified.iter() {
            let mut funcs : Vec<TermApplicationResult> = interpreter_state.get_app_results_with_func(&elem).clone();
            for func in funcs.drain(..) {
                follow_up.insert(func);
            }
        }
        if (follow_up.len() > 0) {
            self.propagate_prior_recursive_helper(interpreter_state, follow_up, all_modified);
        }
    }

    fn propagate_prior_recursive_helper(&mut self, interpreter_state : &InterpreterState,
                                        modified : HashSet::<TermApplicationResult>,
                                        all_modified : &mut HashSet::<TermPointer>) {
        let mut follow_up : HashSet::<TermPointer> = HashSet::new();
        for elem in modified.iter() {
            let ret_ref : TermReference = elem.get_ret_ref();
            if let TermReference::FuncRef(ret_func_ptr) = ret_ref {
                self.propagate_prior(elem.clone());
                follow_up.insert(ret_func_ptr.clone()); 
                all_modified.insert(ret_func_ptr);
            }
        }
        if (follow_up.len() > 0) {
            self.propagate_prior_recursive(interpreter_state, follow_up, all_modified);
        }
    }

    //Propagates data updates upwards
    fn propagate_data_recursive(&mut self, interpreter_state : &InterpreterState, 
                                results : HashSet::<TermApplicationResult>,
                                all_modified : &mut HashSet::<TermPointer>) {
        let mut follow_up : HashSet::<TermPointer> = HashSet::new();
        for elem in results.iter() {
            self.propagate_data(elem.clone());
            let func_ptr : TermPointer = elem.get_func_ptr(); 
            follow_up.insert(func_ptr.clone());
            all_modified.insert(func_ptr);
        }
        if (follow_up.len() > 0) {
            self.propagate_data_recursive_helper(interpreter_state, follow_up, all_modified);
        }
    }

    fn propagate_data_recursive_helper(&mut self, interpreter_state : &InterpreterState, 
                                       modified : HashSet::<TermPointer>,
                                       all_modified : &mut HashSet::<TermPointer>) {
        let mut follow_up : HashSet::<TermApplicationResult> = HashSet::new();
        for elem in modified.iter() {
            let elem_ref = TermReference::FuncRef(elem.clone());

            let mut args : Vec<TermApplicationResult> = interpreter_state.get_app_results_with_arg(&elem_ref).clone();
            for arg in args.drain(..) {
                follow_up.insert(arg);
            }
            let mut rets : Vec<TermApplicationResult> = interpreter_state.get_app_results_with_result(&elem_ref).clone();
            for ret in rets.drain(..) {
                follow_up.insert(ret);
            }
        }
        if (follow_up.len() > 0) {
            self.propagate_data_recursive(interpreter_state, follow_up, all_modified);
        }
    }

    //Given a TermApplicationResult, compute the estimated output from the application
    //and use it to update the model for the result. If an existing update
    //exists for the given application of terms, this will first remove that update
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult) {
        let func_schmear : Schmear = self.get_schmear_from_ptr(&term_app_res.get_func_ptr());
        let arg_schmear : Schmear = self.get_schmear_from_ref(&term_app_res.get_arg_ref());
       
        //Get the model space for the func type
        let func_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_func_type()).unwrap();
        let ret_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_ret_type()).unwrap();

        let out_schmear : Schmear = func_space.apply_schmears(&func_schmear, &arg_schmear);
        let out_prior : NormalInverseGamma = ret_space.schmear_to_prior(&out_schmear);

        if let TermReference::FuncRef(ret_ptr) = term_app_res.get_ret_ref() {
            //Actually perform the update
            let ret_embedding : &mut Model = self.get_mut_embedding(ret_ptr);
            if (ret_embedding.has_prior(&term_app_res.term_app)) {
                ret_embedding.downdate_prior(&term_app_res.term_app);
            }
            ret_embedding.update_prior(term_app_res.term_app, out_prior);
        } else {
            panic!();
        }
    }

    //Given a TermApplicationResult, update the model for the function based on the
    //implicitly-defined data-point for the result
    fn propagate_data(&mut self, term_app_res : TermApplicationResult) {
        let arg_ref = term_app_res.get_arg_ref();
        let ret_ref = term_app_res.get_ret_ref();

        let arg_mean : Array1::<f32> = self.get_mean_from_ref(&arg_ref);
        let out_inv_schmear : InverseSchmear = self.get_inverse_schmear_from_ref(&ret_ref);

        let data_point = DataPoint {
            in_vec : arg_mean,
            out_inv_schmear
        };

        let func_embedding : &mut Model = self.get_mut_embedding(term_app_res.get_func_ptr());
        if (func_embedding.has_data(&arg_ref)) {
            func_embedding.downdate_data(&arg_ref);
        }
        func_embedding.update_data(arg_ref, data_point);
    }

}

