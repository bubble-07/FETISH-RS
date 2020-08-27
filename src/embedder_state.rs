extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::rc::*;
use either::*;
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
use crate::bayes_utils::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::feature_collection::*;
use crate::enum_feature_collection::*;
use crate::vector_space::*;
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
                    let model_sketched_dims = model_space.get_sketched_dimensions();
                    let model_full_dims = model_space.get_full_dimensions();
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

    pub fn thompson_sample_term(&self, type_id : TypeId, inv_schmear : &InverseSchmear) -> (TermPointer, f32) {
        let space : &ModelSpace = self.model_spaces.get(&type_id).unwrap();
        let mut rng = rand::thread_rng();
        let (term_ind, dist) = space.thompson_sample_term(&mut rng, inv_schmear);
        let term_pointer = TermPointer {
            type_id : type_id,
            index : term_ind
        };
        (term_pointer, dist)
    }

    pub fn thompson_sample_app(&self, func_id : TypeId, arg_id : TypeId, target : &InverseSchmear)
                              -> (TermApplication, f32) {
        let func_space : &ModelSpace = self.model_spaces.get(&func_id).unwrap();
        let mut rng = rand::thread_rng();
        //Now we have two cases, depending on whether/not the argument type is a vector type
        if is_vector_type(arg_id) {
            let vec_space = self.vector_spaces.get(&arg_id).unwrap();
            let (func_ind, vec, dist) = func_space.thompson_sample_vec(&mut rng, vec_space, target);
            let func_pointer = TermPointer {
                type_id : func_id,
                index : func_ind
            };
            let vector_ref = TermReference::VecRef(to_noisy(&vec));
            let term_application = TermApplication {
                func_ptr : func_pointer,
                arg_ref : vector_ref
            };
            (term_application, dist) 
        } else {
            let arg_space : &ModelSpace = self.model_spaces.get(&arg_id).unwrap();
            let (func_ind, arg_ind, dist) = func_space.thompson_sample_app(&mut rng, arg_space, target); 
            let func_pointer = TermPointer {
                type_id : func_id,
                index : func_ind
            };
            let arg_pointer = TermPointer {
                type_id : arg_id,
                index : arg_ind
            };
            let term_application = TermApplication {
                func_ptr : func_pointer,
                arg_ref : TermReference::FuncRef(arg_pointer)
            };
            (term_application, dist)
        }
    }

    pub fn find_better_app(&self, term_application : &TermApplication, target : &Array1<f32>) ->
                          (InverseSchmear, Either<InverseSchmear, Array1<R32>>) {
        let func_ptr = &term_application.func_ptr;
        let func_space = self.get_model_space(func_ptr);
        let func_model = self.get_embedding(func_ptr);
        let arg_ref = &term_application.arg_ref;
        match arg_ref {
            TermReference::FuncRef(arg_ptr) => {
                let arg_space = self.get_model_space(arg_ptr);
                let arg_model = self.get_embedding(arg_ptr);
                let arg_inv_schmear = self.get_inverse_schmear_from_ptr(arg_ptr).flatten();
                let compressed_arg_inv_schmear = arg_space.compress_inverse_schmear(&arg_inv_schmear);

                let (func_schmear, arg_schmear) = func_model.find_better_app(compressed_arg_inv_schmear, target);
                let reduced_func_schmear = func_space.compress_inverse_schmear(&func_schmear);

                (reduced_func_schmear, Either::Left(arg_schmear))
            },
            TermReference::VecRef(vec) => {
                let (func_schmear, arg_vec) = func_model.find_better_vec_app(&from_noisy(vec), target);
                let reduced_func_schmear = func_space.compress_inverse_schmear(&func_schmear);
                (reduced_func_schmear, Either::Right(to_noisy(&arg_vec)))
            }
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

    pub fn get_embedding(&self, term_ptr : &TermPointer) -> &Model {
        let space = self.get_model_space(term_ptr);
        space.get_model(term_ptr.index)
    }

    pub fn get_model_space(&self, term_ptr : &TermPointer) -> &ModelSpace {
        self.model_spaces.get(&term_ptr.type_id).unwrap()
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
            TermReference::FuncRef(func_ptr) => self.get_schmear_from_ptr(func_ptr).flatten(),
            TermReference::VecRef(vec) => Schmear::from_vector(&vec)
        }
    }
    fn get_schmear_from_ptr(&self, term_ptr : &TermPointer) -> FuncSchmear {
        let embedding : &Model = self.get_embedding(term_ptr);
        embedding.get_schmear()
    }

    fn get_inverse_schmear_from_ptr(&self, term_ptr : &TermPointer) -> FuncInverseSchmear {
        let embedding : &Model = self.get_embedding(term_ptr);
        embedding.get_inverse_schmear()
    }

    fn get_inverse_schmear_from_ref(&self, term_ref : &TermReference) -> InverseSchmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_inverse_schmear_from_ptr(func_ptr).flatten(),
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
                    topo_sort.insert(application.clone());
                    stack.push(application.clone());
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
                        topo_sort.add_dependency(elem.clone(), application.clone());
                        stack.push(application);
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

    //Given a TermApplicationResult, compute the estimated output from the application
    //and use it to update the model for the result. If an existing update
    //exists for the given application of terms, this will first remove that update
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult) {
        let func_schmear : FuncSchmear = self.get_schmear_from_ptr(&term_app_res.get_func_ptr());
        let full_arg_schmear : Schmear = self.get_schmear_from_ref(&term_app_res.get_arg_ref());
       
        //Get the model space for the func type
        let func_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_func_type()).unwrap();
        let ret_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_ret_type()).unwrap();

        trace!("Propagating prior for space of size {}->{}", func_space.feature_dimensions, 
                                                             func_space.out_dimensions);

        let maybe_arg_space = self.model_spaces.get(&term_app_res.get_arg_type());
        let arg_schmear : Schmear = match (maybe_arg_space) {
            Option::Some(arg_space) => arg_space.compress_schmear(&full_arg_schmear),
            Option::None => full_arg_schmear
        };

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

        let mut arg_mean : Array1::<f32> = self.get_mean_from_ref(&arg_ref);
        let mut out_inv_schmear : InverseSchmear = self.get_inverse_schmear_from_ref(&ret_ref);

        trace!("Propagating data for space of size {}->{}", arg_mean.shape()[0],
                                                            out_inv_schmear.mean.shape()[0]);

        let in_type = term_app_res.get_arg_type();
        if (!is_vector_type(in_type)) {
            let in_space = self.model_spaces.get(&in_type).unwrap();
            arg_mean = in_space.sketch(&arg_mean);  
        }

        let out_type = term_app_res.get_ret_type();
        if (!is_vector_type(out_type)) {
            let out_space = self.model_spaces.get(&out_type).unwrap();
            out_inv_schmear = out_space.compress_inverse_schmear(&out_inv_schmear);
        }

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

