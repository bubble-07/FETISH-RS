extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::sigma_points::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::rc::*;
use crate::data_update::*;
use crate::data_point::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use crate::term_model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::feature_collection::*;
use crate::enum_feature_collection::*;
use crate::vector_space::*;
use crate::normal_inverse_wishart::*;
use topological_sort::TopologicalSort;

extern crate pretty_env_logger;

pub struct EmbedderState {
    pub model_spaces : HashMap::<TypeId, ModelSpace>,
    pub vector_spaces : HashMap::<TypeId, VectorSpace>
}

impl EmbedderState {

    pub fn store_vec(&mut self, vec_type : TypeId, vec : Array1<R32>) {
        self.vector_spaces.get_mut(&vec_type).unwrap().store_vec(vec);
    }

    pub fn new() -> EmbedderState {
        info!("Readying embedder state");
        let mut model_spaces = HashMap::<TypeId, ModelSpace>::new();
        let mut vector_spaces = HashMap::<TypeId, VectorSpace>::new();
        
        let mut full_dimensions = HashMap::<TypeId, usize>::new();
        let mut reduced_dimensions = HashMap::<TypeId, usize>::new();
        let mut topo_sort = TopologicalSort::<TypeId>::new();
        for i in 0..total_num_types() {
            let type_id : TypeId = i as TypeId;
            match get_type(type_id) {
                Type::FuncType(arg_type_id, ret_type_id) => {
                    topo_sort.add_dependency(arg_type_id, type_id);
                    topo_sort.add_dependency(ret_type_id, type_id);
                },
                Type::VecType(dim) => {
                    full_dimensions.insert(type_id, dim);
                    reduced_dimensions.insert(type_id, dim);

                    vector_spaces.insert(type_id, VectorSpace::new(dim));
                }
            };
        }
        
        while (topo_sort.len() > 0) {
            let mut type_ids : Vec<TypeId> = topo_sort.pop_all();
            for func_type_id in type_ids.drain(..) {
                if let Type::FuncType(arg_type_id, ret_type_id) = get_type(func_type_id) {
                    let arg_dimension = *reduced_dimensions.get(&arg_type_id).unwrap();
                    let ret_dimension = *reduced_dimensions.get(&ret_type_id).unwrap();

                    info!("Creating model space with dims {} -> {}", arg_dimension, ret_dimension);
                    let model_space = ModelSpace::new(arg_dimension, ret_dimension);
                    let model_sketched_dims = model_space.space_info.get_sketched_dimensions();
                    let model_full_dims = model_space.space_info.get_full_dimensions();
                    model_spaces.insert(func_type_id, model_space);
                    full_dimensions.insert(func_type_id, model_full_dims);
                    reduced_dimensions.insert(func_type_id, model_sketched_dims);
                }
            }
        }

        EmbedderState {
            model_spaces,
            vector_spaces
        }
    }

    pub fn init_embeddings(&mut self, interpreter_state : &mut InterpreterState) {
        trace!("Initializing embeddings for {} new terms", interpreter_state.new_terms.len());
        for term_ptr in interpreter_state.new_terms.drain(..) {
            if (!self.has_embedding(&term_ptr)) {
                self.init_embedding(term_ptr);
            }
        }
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

    fn get_schmear_from_ref(&self, term_ref : &TermReference) -> Schmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_schmear_from_ptr(func_ptr).flatten(),
            TermReference::VecRef(vec) => Schmear::from_vector(&vec)
        }
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
        let type_id = &term_ptr.type_id;
        let model_space = self.model_spaces.get(type_id).unwrap();
        let full_schmear = self.get_schmear_from_ptr(term_ptr).flatten();
        let result = model_space.space_info.compress_schmear(&full_schmear);
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
            let func_ptr = elem.get_func_ptr();
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
            let to_process = topo_sort.pop_all();
            for elem in to_process {
                self.propagate_prior(elem.clone());
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
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult) {
        let func_schmear = self.get_prior_propagation_func_schmear(&term_app_res);
      
        //Get the model space for the func type
        let func_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_func_type()).unwrap();
        let ret_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_ret_type()).unwrap();

        trace!("Propagating prior for space of size {}->{}", func_space.space_info.feature_dimensions, 
                                                             func_space.space_info.out_dimensions);

        let arg_schmear = self.get_compressed_schmear_from_ref(&term_app_res.get_arg_ref());

        let out_schmear : Schmear = func_space.space_info.apply_schmears(&func_schmear, &arg_schmear);

        if let TermReference::FuncRef(ret_ptr) = term_app_res.get_ret_ref() {
            let out_prior : NormalInverseWishart = ret_space.schmear_to_prior(&self, &ret_ptr, &out_schmear);
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

