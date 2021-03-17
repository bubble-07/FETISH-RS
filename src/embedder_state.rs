extern crate ndarray;
extern crate ndarray_linalg;

use rand::prelude::*;
use crate::space_info::*;
use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::sampled_embedder_state::*;
use crate::data_update::*;
use crate::interpreter_state::*;
use crate::type_id::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application_result::*;
use crate::term_model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::normal_inverse_wishart::*;
use crate::elaborator::*;
use topological_sort::TopologicalSort;

extern crate pretty_env_logger;

pub struct EmbedderState {
    pub model_spaces : HashMap::<TypeId, ModelSpace>,
    pub elaborators : HashMap::<TypeId, Elaborator>
}

impl EmbedderState {

    pub fn sample(&self, rng : &mut ThreadRng) -> SampledEmbedderState {
        let mut embedding_spaces = HashMap::new();
        for (type_id, model_space) in self.model_spaces.iter() {
            let sampled_embedding_space = model_space.sample(rng); 
            embedding_spaces.insert(*type_id, sampled_embedding_space);
        }
        SampledEmbedderState {
            embedding_spaces
        }
    }

    pub fn new() -> EmbedderState {
        info!("Readying embedder state");

        let mut model_spaces = HashMap::<TypeId, ModelSpace>::new();
        let mut elaborators = HashMap::<TypeId, Elaborator>::new();
        for func_type_id in 0..total_num_types() {
            if (!is_vector_type(func_type_id)) {
                let model_space = ModelSpace::new(func_type_id);
                model_spaces.insert(func_type_id, model_space);

                let elaborator = Elaborator::new(func_type_id);
                elaborators.insert(func_type_id, elaborator);
            }
        }

        EmbedderState {
            model_spaces,
            elaborators
        }
    }

    pub fn init_embeddings(&mut self, interpreter_state : &InterpreterState) {
        trace!("Initializing embeddings for {} new terms", interpreter_state.new_terms.len());
        for term_ptr in interpreter_state.new_terms.iter() {
            if (!self.has_embedding(term_ptr)) {
                self.init_embedding(term_ptr.clone());
            }
        }
    }

    pub fn bayesian_update_step(&mut self, interpreter_state : &InterpreterState) {
        self.init_embeddings(interpreter_state);

        let mut data_updated_terms : HashSet<TermPointer> = HashSet::new();
        let mut prior_updated_terms : HashSet<TermPointer> = HashSet::new();

        let mut updated_apps : HashSet::<TermApplicationResult> = HashSet::new();
        for term_app_result in interpreter_state.new_term_app_results.iter() {
            updated_apps.insert(term_app_result.clone()); 
        }

        trace!("Propagating data updates for {} applications", updated_apps.len());
        self.propagate_data_recursive(interpreter_state, updated_apps, &mut data_updated_terms);
        trace!("Propagating prior updates for {} applications", data_updated_terms.len());
        self.propagate_prior_recursive(interpreter_state, data_updated_terms, &mut prior_updated_terms);
    }

    pub fn has_embedding(&self, term_ptr : &TermPointer) -> bool {
        let space : &ModelSpace = self.model_spaces.get(&term_ptr.type_id).unwrap();
        space.has_model(term_ptr.index)
    }

    pub fn get_embedding(&self, term_ptr : &TermPointer) -> &TermModel {
        let space = self.get_model_space(term_ptr);
        space.get_model(term_ptr.index)
    }

    pub fn get_model_space(&self, term_ptr : &TermPointer) -> &ModelSpace {
        self.model_spaces.get(&term_ptr.type_id).unwrap()
    }

    pub fn get_mut_embedding(&mut self, term_ptr : TermPointer) -> &mut TermModel {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.get_model_mut(term_ptr.index)
    }

    pub fn init_embedding(&mut self, term_ptr : TermPointer) {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.add_model(term_ptr.index)
    }

    fn get_schmear_from_ptr(&self, term_ptr : &TermPointer) -> FuncSchmear {
        let embedding : &TermModel = self.get_embedding(term_ptr);
        embedding.get_schmear()
    }

    fn get_inverse_schmear_from_ptr(&self, term_ptr : &TermPointer) -> FuncInverseSchmear {
        let embedding : &TermModel = self.get_embedding(term_ptr);
        embedding.get_inverse_schmear()
    }

    fn get_compressed_schmear_from_ptr(&self, term_ptr : &TermPointer) -> Schmear {
        let type_id = term_ptr.type_id;
        let func_schmear = self.get_schmear_from_ptr(term_ptr);
        let func_feat_info = get_feature_space_info(type_id);
        let projection_mat = func_feat_info.get_projection_matrix();
        let result = func_schmear.compress(&projection_mat);
        result
    }

    fn get_compressed_schmear_from_ref(&self, term_ref : &TermReference) -> Schmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_compressed_schmear_from_ptr(&func_ptr),
            TermReference::VecRef(vec) => Schmear::from_vector(&vec)
        }
    }

    //Propagates prior updates downwards
    pub fn propagate_prior_recursive(&mut self, interpreter_state : &InterpreterState,
                                     to_propagate : HashSet::<TermPointer>,
                                     all_modified : &mut HashSet::<TermPointer>) {
        let mut topo_sort = TopologicalSort::<TermApplicationResult>::new();
        let mut stack = Vec::<TermApplicationResult>::new();

        info!("Obtaining elaborator func schmears");
        let mut elaborator_func_schmears = HashMap::new();
        for type_id in 0..total_num_types() {
            if (!is_vector_type(type_id)) {
                let elaborator = self.elaborators.get(&type_id).unwrap(); 
                let elaborator_func_schmear = elaborator.get_expansion_func_schmear();
                elaborator_func_schmears.insert(type_id, elaborator_func_schmear);
            }
        }
        info!("Propagating priors");

        for func_ptr in to_propagate {
            let applications = interpreter_state.get_app_results_with_func(&func_ptr);
            for application in applications {
                if let TermReference::FuncRef(_) = application.get_ret_ref() {
                    if (self.has_nontrivial_prior_update(&application)) {
                        topo_sort.insert(application.clone());
                        stack.push(application.clone());
                    }
                }
            }
        }
        while (stack.len() > 0) {
            let elem = stack.pop().unwrap();
            let ret_ref = elem.get_ret_ref();
            if let TermReference::FuncRef(ret_func_ptr) = ret_ref {
                let applications = interpreter_state.get_app_results_with_func(&ret_func_ptr); 
                for application in applications {
                    if let TermReference::FuncRef(_) = application.get_ret_ref() {
                        if (self.has_nontrivial_prior_update(&application)) {
                            topo_sort.add_dependency(elem.clone(), application.clone());
                            stack.push(application);
                        }
                    }
                }

                all_modified.insert(ret_func_ptr);
            }
        }

        while (!topo_sort.is_empty()) {
            let mut to_process = topo_sort.pop_all();
            for elem in to_process.drain(..) {
                let out_type = elem.get_ret_type();
                let elaborator_func_schmear = elaborator_func_schmears.get(&out_type).unwrap();
                self.propagate_prior(elem, elaborator_func_schmear);
            }
        }
    }

    //Propagates data updates upwards
    pub fn propagate_data_recursive(&mut self, interpreter_state : &InterpreterState,
                                    to_propagate : HashSet::<TermApplicationResult>,
                                    all_modified : &mut HashSet::<TermPointer>) {
        let mut topo_sort = TopologicalSort::<TermApplicationResult>::new();
        let mut stack = Vec::<TermApplicationResult>::new();

        for elem in to_propagate {
            stack.push(elem.clone());
        }

        while (stack.len() > 0) {
            let elem = stack.pop().unwrap();
            let func_ptr = elem.get_func_ptr();
            let func_ref = TermReference::FuncRef(func_ptr.clone());

            all_modified.insert(func_ptr);

            let args = interpreter_state.get_app_results_with_arg(&func_ref);
            for arg in args {
                stack.push(arg.clone());
                topo_sort.add_dependency(elem.clone(), arg.clone());
            }

            let rets = interpreter_state.get_app_results_with_result(&func_ref);
            for ret in rets {
                stack.push(ret.clone());
                topo_sort.add_dependency(elem.clone(), ret.clone());
            }

            topo_sort.insert(elem);
        }

        while (!topo_sort.is_empty()) {
            let to_process = topo_sort.pop_all();
            for elem in to_process {
                self.propagate_data(elem);
            }
        }
        
    }

    fn get_prior_propagation_func_schmear(&self, term_app_res : &TermApplicationResult) -> FuncSchmear {
        let func_model = self.get_embedding(&term_app_res.get_func_ptr());
        //If the model for the function has any data update involving
        //the argument, we need to consider the model with it removed
        let arg_ref = &term_app_res.get_arg_ref();
        let func_schmear = if (func_model.has_data(arg_ref)) {
            let data_update = func_model.data_updates.get(arg_ref).unwrap();
            let mut downdated_distr = func_model.model.data.clone();
            downdated_distr -= data_update;

            downdated_distr.get_schmear()
        } else {
            func_model.get_schmear()
        };
        func_schmear 
    }

    fn has_nontrivial_prior_update(&self, term_app_res : &TermApplicationResult) -> bool {
        let func_model = self.get_embedding(&term_app_res.get_func_ptr());
        let arg_ref = &term_app_res.get_arg_ref();
        let mut num_data_updates = func_model.data_updates.len();
        if (func_model.has_data(arg_ref)) {
            num_data_updates -= 1;
        }
        num_data_updates > 0
    }

    //Given a TermApplicationResult, compute the estimated output from the application
    //and use it to update the model for the result. If an existing update
    //exists for the given application of terms, this will first remove that update
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult,
                       elaborator_func_schmear : &FuncSchmear) {
        let func_schmear = self.get_prior_propagation_func_schmear(&term_app_res);
      
        //Get the model space for the func type
        let ret_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_ret_type()).unwrap();

        let func_space_info = get_function_space_info(term_app_res.get_func_type());

        trace!("Propagating prior for space of size {}->{}", func_space_info.get_feature_dimensions(), 
                                                             func_space_info.get_output_dimensions());

        let arg_schmear = self.get_compressed_schmear_from_ref(&term_app_res.get_arg_ref());

        let out_schmear : Schmear = func_space_info.apply_schmears(&func_schmear, &arg_schmear);

        if let TermReference::FuncRef(ret_ptr) = term_app_res.get_ret_ref() {
            let out_prior : NormalInverseWishart = ret_space.schmear_to_prior(&self, elaborator_func_schmear,
                                                                              &ret_ptr, &out_schmear);
            //Actually perform the update
            let ret_embedding : &mut TermModel = self.get_mut_embedding(ret_ptr);
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

        let arg_schmear = self.get_compressed_schmear_from_ref(&arg_ref);
        let ret_schmear = self.get_compressed_schmear_from_ref(&ret_ref);

        let arg_mean : Array1::<f32> = arg_schmear.mean;

        trace!("Propagating data for space of size {}->{}", arg_mean.shape()[0],
                                                            ret_schmear.mean.shape()[0]);

        let out_type = term_app_res.get_ret_type();
        let data_update = if (is_vector_type(out_type)) {
            let ret_mean = ret_schmear.mean;
            construct_vector_data_update(arg_mean, ret_mean)
        } else {
            construct_data_update(arg_mean, &ret_schmear)
        };

        let func_embedding : &mut TermModel = self.get_mut_embedding(term_app_res.get_func_ptr());
        if (func_embedding.has_data(&arg_ref)) {
            func_embedding.downdate_data(&arg_ref);
        }
        func_embedding.update_data(arg_ref, data_update);
    }

}

