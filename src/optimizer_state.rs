extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use rand::prelude::*;
use crate::array_utils::*;
use crate::typed_vector::*;
use crate::sampled_embedder_state::*;
use crate::type_action::*;
use crate::term_model::*;
use crate::linalg_utils::*;
use crate::application_chain::*;
use crate::type_id::*;
use crate::type_graph::*;
use crate::space_info::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::data_point::*;
use crate::interpreter_and_embedder_state::*;
use crate::model::*;
use crate::inverse_schmear::*;
use crate::schmeared_hole::*;

use crate::value_field_state::*;
use crate::function_optimum_state::*;

extern crate pretty_env_logger;

pub struct OptimizerState {
    pub interpreter_and_embedder_state : InterpreterAndEmbedderState,
    pub value_field_state : ValueFieldState,
    pub func_opt_state : FunctionOptimumState,
    pub target_type_id : TypeId,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>
}

impl OptimizerState {

    pub fn get_target_type_id(&self) -> TypeId {
        self.target_type_id
    }

    pub fn step(&mut self) -> Option<TermPointer> {
        trace!("Applying Bayesian update");
        self.interpreter_and_embedder_state.bayesian_update_step();

        trace!("Sampling embedder state");
        let mut rng = rand::thread_rng();
        let sampled_embedder_state = self.interpreter_and_embedder_state.embedder_state.sample(&mut rng);

        trace!("Updating value fields");
        trace!("Obtaining new constraints");
        let mut constraints = self.interpreter_and_embedder_state.get_new_constraints(&sampled_embedder_state);
        constraints.update_repeat(NUM_CONSTRAINT_REPEATS);
        constraints.update_shuffle();

        trace!("Applying new constraints");
        self.value_field_state.apply_constraints(&constraints);

        trace!("Optimizing for highest-value application chain");
        let best_application_chain = self.find_best_application_chain(&sampled_embedder_state);

        trace!("Done with newly-evaluated terms, discarding");
        self.interpreter_and_embedder_state.clear_newly_received();

        trace!("Evaluating best application");
        let result_ref = self.interpreter_and_embedder_state
                             .interpreter_state.evaluate_application_chain(&best_application_chain);

        if (result_ref.get_type() == self.get_target_type_id()) {
            trace!("Best term was in the target type. Evaluating on training data");
            if let TermReference::FuncRef(func_ptr) = &result_ref {
                self.evaluate_training_data_step(func_ptr.clone());
            }
        }

        match (result_ref) {
            TermReference::FuncRef(func_ptr) => Option::Some(func_ptr),
            TermReference::VecRef(_) => Option::None
        }
    }

    pub fn find_best_next_with_transition(&self, sampled_embedder_state : &SampledEmbedderState,
                                      current_compressed_vec : &TypedVector, 
                                      transition : &TypeAction) ->
                                      (TermReference, TypedVector, f32) {

        //The current type id must be a function type
        match (transition) {
            TypeAction::Applying(func_type_id) => {
                //In this case, both the function and the argument will be functions
                let (func_ptr, next_compressed_vec, value) = 
                    sampled_embedder_state.get_best_term_to_apply(current_compressed_vec, *func_type_id,
                                                                  &self.value_field_state);
                let func_ref = TermReference::FuncRef(func_ptr); 
                (func_ref, next_compressed_vec, value)
            },
            TypeAction::Passing(arg_type_id) => {
                if (is_vector_type(*arg_type_id)) {
                    let (arg_vec, next_compressed_vec, value) = 
                        self.func_opt_state.get_best_vector_to_pass(current_compressed_vec,
                                                    &self.value_field_state, sampled_embedder_state);
                    let arg_ref = TermReference::VecRef(to_noisy(&arg_vec));
                    (arg_ref, next_compressed_vec, value)
                } else {
                    let (arg_ptr, next_compressed_vec, value) =
                        sampled_embedder_state.get_best_term_to_pass(current_compressed_vec, 
                                                                     &self.value_field_state);
                    let arg_ref = TermReference::FuncRef(arg_ptr);
                    (arg_ref, next_compressed_vec, value)
                }
            }
        }
    }

    pub fn find_best_next_application(&self, sampled_embedder_state : &SampledEmbedderState,
                                      current_compressed_vec : &TypedVector) ->
                                      (TermReference, TypedVector, f32) {
        let current_type_id = current_compressed_vec.type_id;
        let target_type_id = self.get_target_type_id();

        let mut best_value = f32::NEG_INFINITY;
        let mut best_term = Option::None;
        let mut best_vec = Option::None;

        let successor_types =  get_type_successors(current_type_id);
        for successor_type in successor_types.iter() {
            if is_type_reachable_from(*successor_type, target_type_id) {
                let type_actions = get_type_actions(current_type_id, *successor_type);
                for type_action in type_actions.iter() {
                    let (term, vec, value) = self.find_best_next_with_transition(sampled_embedder_state,
                                                  current_compressed_vec, type_action);
                    if (value > best_value) {
                        best_term = Option::Some(term.clone());
                        best_vec = Option::Some(vec.clone());
                        best_value = value;
                    }
                }
            }
        }
        (best_term.unwrap(), best_vec.unwrap(), best_value)
    }

    pub fn find_playout_next_application(&self, sampled_embedder_state : &SampledEmbedderState,
                                         current_compressed_vec : &TypedVector) ->
                                       (TermReference, TypedVector) {
        let current_type_id = current_compressed_vec.type_id;
        let target_type_id = self.get_target_type_id();

        let successor_types = get_type_successors(current_type_id);
    
        let mut rng = rand::thread_rng();

        let mut random_usize : usize = rng.gen();
        let mut successor_ind = random_usize % successor_types.len();
        while (!is_type_reachable_from(successor_types[successor_ind], target_type_id)) {
            random_usize = rng.gen();
            successor_ind = random_usize % successor_types.len();
        }

        let successor_type = successor_types[successor_ind];

        let type_actions = get_type_actions(current_type_id, successor_type);

        random_usize = rng.gen();
        let action_ind = random_usize & type_actions.len();

        let action = &type_actions[action_ind];

        let (term, vec, _) = self.find_best_next_with_transition(sampled_embedder_state, 
                                                                current_compressed_vec,
                                                                action);

        (term, vec)
    }
    
    pub fn find_best_application_chain(&mut self, 
                                       sampled_embedder_state : &SampledEmbedderState) -> ApplicationChain {
        let target_type_id = self.get_target_type_id();

        let best_application = self.find_best_application(sampled_embedder_state); 

        let mut current_compressed_vec = sampled_embedder_state.evaluate_term_application(&best_application);
        let mut current_chain = ApplicationChain::from_term_application(best_application);

        let mut current_best_chain = Option::None;
        let mut current_best_value = f32::NEG_INFINITY;

        for _ in 0..OPT_MAX_ITERS {
            //Pick best actions until we get to the target type, then compare with current best
            //chain
            let (picked_term, next_compressed_vec, current_value) = 
                self.find_best_next_application(sampled_embedder_state, &current_compressed_vec);

            current_compressed_vec = next_compressed_vec;
            current_chain.add_to_chain(picked_term);

            if (current_compressed_vec.type_id == target_type_id) {
                if (current_value > current_best_value) {
                    current_best_value = current_value;
                    current_best_chain = Option::Some(current_chain.clone());
                } else {
                    //If we've hit the target again, but the chain
                    //has gotten worst, fall back on the current best chain
                    return current_best_chain.unwrap();
                }
            }
        }
        
        while (current_compressed_vec.type_id != target_type_id) { 
            //Pick playout actions until we get to the target type, then yield that
            let (picked_term, next_compressed_vec) = 
                self.find_playout_next_application(sampled_embedder_state, &current_compressed_vec);

            current_compressed_vec = next_compressed_vec;
            current_chain.add_to_chain(picked_term);
        }
        current_chain
    }

    pub fn find_best_application(&mut self, sampled_embedder_state : &SampledEmbedderState) -> TermApplication {
        let (nonvec_app, nonvec_value) = 
            sampled_embedder_state.get_best_nonvector_application_with_value(&self.interpreter_and_embedder_state.interpreter_state,
                                                                             &self.value_field_state);
        let (vec_app, vec_value) = self.func_opt_state.update(sampled_embedder_state, &self.value_field_state);
        if (vec_value > nonvec_value) {
            vec_app
        } else {
            nonvec_app
        }
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        self.interpreter_and_embedder_state.evaluate_training_data_step(term_ptr, &self.data_points);
    }

    pub fn init_step(&mut self) {
        self.interpreter_and_embedder_state.init_step();
    }

    pub fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>) -> OptimizerState {

        //Step 1: find the embedding of the target term

        if (data_points.is_empty()) {
            panic!(); 
        }
        info!("Readying types");
        let in_dimensions : usize = data_points[0].0.shape()[0];
        let out_dimensions : usize = data_points[0].1.shape()[0];

        let in_type_id : TypeId = get_type_id(&Type::VecType(in_dimensions));
        let out_type_id : TypeId = get_type_id(&Type::VecType(out_dimensions));

        info!("Readying interpreter state");
        let interpreter_and_embedder_state = InterpreterAndEmbedderState::new();

        info!("Readying target");

        let prior_specification = TermModelPriorSpecification { };
        
        let mut target_model : Model = Model::new(&prior_specification, in_type_id, out_type_id);

        for (in_vec, out_vec) in data_points.iter() {
            let data_point = DataPoint {
                in_vec : in_vec.clone(),
                out_vec : out_vec.clone(),
                weight : 1.0f32
            };
            target_model += data_point;
        }

        let target = target_model.get_schmeared_hole().rescale_spread(TARGET_INV_SCHMEAR_SCALE_FAC);
        let target_type_id = target.type_id;

        let value_field_state = ValueFieldState::new(target);

        let func_opt_state = FunctionOptimumState::new();

        OptimizerState {
            interpreter_and_embedder_state,
            value_field_state,
            func_opt_state,
            target_type_id,
            data_points
        }
    }
}
