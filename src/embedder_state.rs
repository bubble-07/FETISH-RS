extern crate ndarray;
extern crate ndarray_linalg;

use crate::multiple::*;
use rand::prelude::*;
use crate::prior_specification::*;
use crate::space_info::*;
use crate::newly_evaluated_terms::*;
use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::input_to_schmeared_output::*;
use crate::sampled_embedder_state::*;
use crate::term_index::*;
use crate::interpreter_state::*;
use crate::type_id::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application_result::*;
use crate::term_model::*;
use crate::embedding_space::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::normal_inverse_wishart::*;
use crate::elaborator::*;
use topological_sort::TopologicalSort;
use crate::context::*;
use serde::{Serialize, Deserialize};

///An [`EmbedderState`] keeps track of the embeddings of function terms ([`TermModel`]s)
///which come from some [`InterpreterState`], and also the learned [`Elaborator`]s for
///every function type.
pub struct EmbedderState<'a> {
    pub model_spaces : HashMap::<TypeId, EmbeddingSpace<'a>>,
    pub ctxt : &'a Context
}

#[derive(Serialize, Deserialize)]
pub struct SerializedEmbedderState {
    pub model_spaces : HashMap::<TypeId, SerializedEmbeddingSpace>
}

impl SerializedEmbedderState {
    pub fn deserialize<'a>(mut self, ctxt : &'a Context) -> EmbedderState<'a> {
        let mut model_spaces = HashMap::new(); 
        for (type_id, serialized_embedding_space) in self.model_spaces.drain() {
            let embedding_space = serialized_embedding_space.deserialize(ctxt);
            model_spaces.insert(type_id, embedding_space);
        }
        EmbedderState {
            model_spaces,
            ctxt
        }
    }
}

impl<'a> EmbedderState<'a> {
    pub fn serialize(mut self) -> SerializedEmbedderState {
        let mut model_spaces = HashMap::new(); 
        for (type_id, deserialized_embedding_space) in self.model_spaces.drain() {
            let embedding_space = deserialized_embedding_space.serialize();
            model_spaces.insert(type_id, embedding_space);
        }
        SerializedEmbedderState {
            model_spaces
        }
    }

    ///Draws a sample from the distribution over [`TermModel`]s represented in this
    ///[`EmbedderState`], yielding a [`SampledEmbedderState`].
    pub fn sample(&self, rng : &mut ThreadRng) -> SampledEmbedderState<'a> {
        let mut embedding_spaces = HashMap::new();
        for (type_id, model_space) in self.model_spaces.iter() {
            let sampled_embedding_space = model_space.sample(rng); 
            embedding_spaces.insert(*type_id, sampled_embedding_space);
        }
        SampledEmbedderState {
            embedding_spaces,
            ctxt : self.ctxt
        }
    }

    ///Creates a new [`EmbedderState`], initially populated with default embeddings
    ///for primitive terms in the passed [`Context`].
    pub fn new(ctxt : &'a Context) -> EmbedderState<'a> {
        info!("Readying embedder state");

        let mut model_spaces = HashMap::new();
        for func_type_id in 0..ctxt.get_total_num_types() {
            if (!ctxt.is_vector_type(func_type_id)) {
                let mut model_space = EmbeddingSpace::new(func_type_id, ctxt);

                //Initialize embeddings for primitive terms
                let primitive_type_space = ctxt.primitive_directory
                                               .primitive_type_spaces.get(&func_type_id).unwrap();
                for term_index in 0..primitive_type_space.terms.len() {
                    model_space.add_model(TermIndex::Primitive(term_index));
                }

                model_spaces.insert(func_type_id, model_space);
            }
        }

        EmbedderState {
            model_spaces,
            ctxt
        }
    }

    ///Initializes default embeddings for the passed collection of terms in a
    ///[`NewlyEvaluatedTerms`].
    pub fn init_embeddings_for_new_terms(&mut self, newly_evaluated_terms : &NewlyEvaluatedTerms) {
        trace!("Initializing embeddings for {} new terms", newly_evaluated_terms.terms.len());
        for nonprimitive_term_ptr in newly_evaluated_terms.terms.iter() {
            let term_ptr = TermPointer::from(nonprimitive_term_ptr.clone());
            if (!self.has_embedding(term_ptr)) {
                self.init_embedding(term_ptr);
            }
        }
    }

    ///Given an [`InterpreterState`] and a collection of [`NewlyEvaluatedTerms`], performs
    ///a bottom-up (data) update followed by a top-down (prior) update recursively
    ///on all modified terms. This method may be used to keep the [`TermModel`]s in this
    ///[`EmbedderState`] up-to-date with new information.
    pub fn bayesian_update_step(&mut self, interpreter_state : &InterpreterState,
                                           newly_evaluated_terms : &NewlyEvaluatedTerms) {
        self.init_embeddings_for_new_terms(newly_evaluated_terms);

        let mut data_updated_terms : HashSet<TermPointer> = HashSet::new();
        let mut prior_updated_terms : HashSet<TermPointer> = HashSet::new();

        let mut updated_apps : HashSet::<TermApplicationResult> = HashSet::new();
        for term_app_result in newly_evaluated_terms.term_app_results.iter() {
            updated_apps.insert(term_app_result.clone()); 
        }

        trace!("Propagating data updates for {} applications", updated_apps.len());
        self.propagate_data_recursive(interpreter_state, &updated_apps, &mut data_updated_terms,
                                      newly_evaluated_terms);
        trace!("Propagating prior updates for {} applications", data_updated_terms.len());
        self.propagate_prior_recursive(interpreter_state, &data_updated_terms, &mut prior_updated_terms,
                                       newly_evaluated_terms);

        let mut all_updated_terms = HashSet::new();
        for data_updated_term in data_updated_terms.drain() {
            all_updated_terms.insert(data_updated_term);
        }
        for prior_updated_term in prior_updated_terms.drain() {
            all_updated_terms.insert(prior_updated_term);
        }
        self.update_elaborators(all_updated_terms);
    }

    ///Determines whether/not there is a stored [`TermModel`] for the given
    ///[`TermPointer`].
    pub fn has_embedding(&self, term_ptr : TermPointer) -> bool {
        let space : &EmbeddingSpace = self.model_spaces.get(&term_ptr.type_id).unwrap();
        space.has_model(term_ptr.index)
    }

    ///Given a [`TermPointer`] pointing to a [`TermModel`] tracked by this
    ///[`EmbedderState`], yields a reference to the [`TermModel`]. Panics if there is
    ///no such entry stored.
    pub fn get_embedding(&self, term_ptr : TermPointer) -> &TermModel {
        let space = self.get_model_space(term_ptr.type_id);
        space.get_model(term_ptr.index)
    }

    fn get_model_space(&self, type_id : TypeId) -> &EmbeddingSpace {
        self.model_spaces.get(&type_id).unwrap()
    }

    ///Like [`EmbedderState#get_embedding`], but yields a mutable reference to the
    ///[`TermModel`] given a [`TermPointer`] pointing to it. 
    pub fn get_mut_embedding(&mut self, term_ptr : TermPointer) -> &mut TermModel<'a> {
        let space : &mut EmbeddingSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.get_model_mut(term_ptr.index)
    }

    fn init_embedding(&mut self, term_ptr : TermPointer) {
        let space : &mut EmbeddingSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.add_model(term_ptr.index)
    }

    fn get_schmear_from_ptr(&self, term_ptr : TermPointer) -> FuncSchmear {
        let embedding : &TermModel = self.get_embedding(term_ptr);
        embedding.get_schmear()
    }

    fn get_inverse_schmear_from_ptr(&self, term_ptr : TermPointer) -> FuncInverseSchmear {
        let embedding : &TermModel = self.get_embedding(term_ptr);
        embedding.get_inverse_schmear()
    }

    fn get_compressed_schmear_from_ptr(&self, term_ptr : TermPointer) -> Schmear {
        let type_id = term_ptr.type_id;
        let func_schmear = self.get_schmear_from_ptr(term_ptr);
        let func_feat_info = self.ctxt.get_feature_space_info(type_id);
        let projection_mat = func_feat_info.get_projection_matrix();
        let result = func_schmear.compress(projection_mat.view());
        result
    }

    fn get_compressed_schmear_from_ref(&self, term_ref : &TermReference) -> Schmear {
        match term_ref {
            TermReference::FuncRef(func_ptr) => self.get_compressed_schmear_from_ptr(*func_ptr),
            TermReference::VecRef(_, vec) => Schmear::from_vector(vec.view())
        }
    }

    fn update_elaborators(&mut self, mut updated_terms : HashSet::<TermPointer>) {
        for term_ptr in updated_terms.drain() {
            let model_space = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
            let elaborator = &mut model_space.elaborator;

            //Remove existing data for the term
            if (elaborator.has_data(&term_ptr.index)) {
                elaborator.downdate_data(&term_ptr.index);
            }

            let term_model = model_space.models.get(&term_ptr.index).unwrap();
            elaborator.update_data(term_ptr.index, &term_model.model);
        }
    }

    //Propagates prior updates downwards
    fn propagate_prior_recursive(&mut self, interpreter_state : &InterpreterState,
                                     to_propagate : &HashSet::<TermPointer>,
                                     all_modified : &mut HashSet::<TermPointer>,
                                     newly_evaluated : &NewlyEvaluatedTerms) {
        let new_count_map = newly_evaluated.get_count_map();

        let mut topo_sort = TopologicalSort::<TermApplicationResult>::new();
        let mut stack = Vec::<TermApplicationResult>::new();

        for func_ptr in to_propagate {
            let applications = interpreter_state.get_app_results_with_func(*func_ptr);
            for application in applications {
                if let TermReference::FuncRef(_) = application.get_ret_ref() {
                    if (self.has_nontrivial_prior_update(&application)) {
                        topo_sort.insert(application.clone());
                        stack.push(application.clone());
                    }
                }
            }
        }

        let mut ret_type_set = HashSet::new();
        while (stack.len() > 0) {
            let elem = stack.pop().unwrap();
            let ret_ref = elem.get_ret_ref();

            ret_type_set.insert(elem.get_ret_type(self.ctxt));

            if let TermReference::FuncRef(ret_func_ptr) = ret_ref {
                let applications = interpreter_state.get_app_results_with_func(ret_func_ptr); 
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

        info!("Obtaining elaborator func schmears");
        let mut elaborator_func_schmears = HashMap::new();
        for type_id in ret_type_set.drain() {
            if (!self.ctxt.is_vector_type(type_id)) {
                let model_space = self.model_spaces.get(&type_id).unwrap();
                let elaborator = &model_space.elaborator;
                let elaborator_func_schmear = elaborator.get_expansion_func_schmear();
                elaborator_func_schmears.insert(type_id, elaborator_func_schmear);
            }
        }
        info!("Propagating priors");

        while (!topo_sort.is_empty()) {
            let mut to_process = topo_sort.pop_all();
            for elem in to_process.drain(..) {
                let out_type = elem.get_ret_type(self.ctxt);
                let elaborator_func_schmear = elaborator_func_schmears.get(&out_type).unwrap();

                let new_count = match (new_count_map.get(&elem)) {
                    Option::None => 0,
                    Option::Some(count) => *count
                };

                self.propagate_prior(elem, elaborator_func_schmear, new_count);
            }
        }
    }

    //Propagates data updates upwards
    fn propagate_data_recursive(&mut self, interpreter_state : &InterpreterState,
                                    to_propagate : &HashSet::<TermApplicationResult>,
                                    all_modified : &mut HashSet::<TermPointer>,
                                    newly_evaluated : &NewlyEvaluatedTerms) {
        let new_count_map = newly_evaluated.get_count_map();

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
                let new_count = match (new_count_map.get(&elem)) {
                    Option::None => 0,
                    Option::Some(count) => *count
                };
                self.propagate_data(elem, new_count);
            }
        }
        
    }

    fn get_prior_propagation_func_schmear(&self, term_app_res : &TermApplicationResult) -> FuncSchmear {
        let func_model = self.get_embedding(term_app_res.get_func_ptr());
        //If the model for the function has data updates involving this
        //exact same [`TermApplicationResult`] (which it should), we need to remove all data which
        //was added to the model for it, or we risk re-inforcing redundant information.
        let term_input_output = term_app_res.get_term_input_output();

        let mut model_clone = func_model.clone();
        model_clone.downdate_data(&term_input_output);
        model_clone.get_schmear()
    }

    fn has_nontrivial_prior_update(&self, term_app_res : &TermApplicationResult) -> bool {
        let term_input_output = term_app_res.get_term_input_output();
        let func_model = self.get_embedding(term_app_res.get_func_ptr());
        func_model.has_some_data_other_than(&term_input_output)
    }

    //Given a TermApplicationResult, compute the estimated output from the application
    //and use it to update the model for the result. If an existing update
    //exists for the given application of terms, this will first remove that update
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult,
                       elaborator_func_schmear : &FuncSchmear, count_increment : usize) {
        let func_schmear = self.get_prior_propagation_func_schmear(&term_app_res);
      
        //Get the model space for the func type
        let ret_space : &EmbeddingSpace = self.model_spaces.get(&term_app_res.get_ret_type(self.ctxt)).unwrap();

        let func_space_info = self.ctxt.get_function_space_info(term_app_res.get_func_type());

        trace!("Propagating prior for space of size {}->{}", func_space_info.get_feature_dimensions(), 
                                                             func_space_info.get_output_dimensions());

        let arg_schmear = self.get_compressed_schmear_from_ref(&term_app_res.get_arg_ref());

        let out_schmear : Schmear = func_space_info.apply_schmears(&func_schmear, &arg_schmear);

        if let TermReference::FuncRef(ret_ptr) = term_app_res.get_ret_ref() {
            let out_prior : NormalInverseWishart = ret_space.schmear_to_prior(&self, elaborator_func_schmear,
                                                                              ret_ptr, &out_schmear);
            //Actually perform the update
            let ret_embedding : &mut TermModel = self.get_mut_embedding(ret_ptr);
            let prev_count = ret_embedding.downdate_prior(&term_app_res.term_app);
            let new_count = prev_count + count_increment;

            let out_update = Multiple {
                elem : out_prior,
                count : new_count
            };
            ret_embedding.update_prior(term_app_res.term_app, out_update);
        } else {
            panic!();
        }
    }

    //Given a TermApplicationResult, update the model for the function based on the
    //implicitly-defined data-point for the result
    fn propagate_data(&mut self, term_app_res : TermApplicationResult, count_increment : usize) {
        let term_input_output = term_app_res.get_term_input_output();
        let arg_ref = term_app_res.get_arg_ref();
        let ret_ref = term_app_res.get_ret_ref();

        let arg_schmear = self.get_compressed_schmear_from_ref(&arg_ref);
        let ret_schmear = self.get_compressed_schmear_from_ref(&ret_ref);

        let arg_mean : Array1::<f32> = arg_schmear.mean;

        trace!("Propagating data for space of size {}->{}", arg_mean.shape()[0],
                                                            ret_schmear.mean.shape()[0]);

        let data_point = InputToSchmearedOutput {
            in_vec : arg_mean,
            out_schmear : ret_schmear 
        };

        let func_embedding : &mut TermModel = self.get_mut_embedding(term_app_res.get_func_ptr());
        let prev_count = func_embedding.downdate_data(&term_input_output);
        let new_count = prev_count + count_increment;

        let data_update = Multiple {
            elem : data_point,
            count : new_count
        };
        func_embedding.update_data(term_input_output, data_update);
    }
}

