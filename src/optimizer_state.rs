extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use rand::prelude::*;
use fetish_lib::everything::*;
use crate::sampled_value_field_state::*;
use crate::params::*;
use crate::type_action::*;
use crate::application_chain::*;
use crate::constraint_collection::*;

use crate::interpreter_and_embedder_state::*;
use crate::value_field_state::*;
use crate::function_optimum_state::*;
use crate::type_graph::*;
use crate::sampled_embedder_state::*;
use crate::term_model::*;
use crate::elaborator::*;

extern crate pretty_env_logger;

pub struct OptimizerState<'a> {
    pub interpreter_and_embedder_state : InterpreterAndEmbedderState<'a>,
    pub value_field_state : ValueFieldState<'a>,
    pub func_opt_state : FunctionOptimumState<'a>,
    pub target_type_id : TypeId,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>,
    pub type_graph : TypeGraph<'a>
}

impl <'a> OptimizerState<'a> {
    fn get_context(&self) -> &Context {
        self.interpreter_and_embedder_state.get_context()
    }

    pub fn get_target_type_id(&self) -> TypeId {
        self.target_type_id
    }

    pub fn step(&mut self) -> Option<TermPointer> {
        trace!("Applying Bayesian update");
        self.interpreter_and_embedder_state.bayesian_update_step();

        trace!("Sampling embedder state");
        let mut rng = rand::thread_rng();
        let sampled_embedder_state = self.interpreter_and_embedder_state.embedder_state.sample(&mut rng);

        let newly_evaluated_terms = &self.interpreter_and_embedder_state.newly_evaluated_terms;

        trace!("Updating value fields");
        trace!("Obtaining new constraints");
        let mut constraints = get_new_constraints(&sampled_embedder_state, newly_evaluated_terms);
        constraints.update_repeat(NUM_CONSTRAINT_REPEATS);
        constraints.update_shuffle();

        trace!("Sampling value field state");
        let mut sampled_value_field_state = self.value_field_state.sample(&sampled_embedder_state);

        trace!("Applying new constraints");
        sampled_value_field_state.apply_constraints(&constraints);

        trace!("Optimizing for highest-value application chain");
        let best_application_chain = self.find_best_application_chain(&sampled_embedder_state, 
                                                                      &sampled_value_field_state);

        trace!("Done with newly-evaluated terms, discarding");
        self.interpreter_and_embedder_state.clear_newly_received();

        trace!("Evaluating best application");
        let result_ref = evaluate_application_chain(&mut self.interpreter_and_embedder_state, &best_application_chain);

        if (result_ref.get_type() == self.get_target_type_id()) {
            trace!("Best term was in the target type. Evaluating on training data");
            if let TermReference::FuncRef(func_ptr) = &result_ref {
                self.evaluate_training_data_step(func_ptr.clone());
            }
        }

        match (result_ref) {
            TermReference::FuncRef(func_ptr) => Option::Some(func_ptr),
            TermReference::VecRef(_, _) => Option::None
        }
    }

    pub fn find_best_next_with_transition(&self, sampled_embedder_state : &SampledEmbedderState,
                                      sampled_value_field_state : &SampledValueFieldState,
                                      current_compressed_vec : &TypedVector, 
                                      transition : &TypeAction) ->
                                      Option<(TermReference, TypedVector, f32)> {

        //The current type id must be a function type
        match (transition) {
            TypeAction::Applying(func_type_id) => {
                //In this case, both the function and the argument will be functions
                let maybe_result = 
                    get_best_term_to_apply(sampled_embedder_state, current_compressed_vec, *func_type_id,
                                                                  sampled_value_field_state);
                if (maybe_result.is_none()) {
                    return Option::None
                }
                let (func_ptr, next_compressed_vec, value) = maybe_result.unwrap();
                let func_ref = TermReference::FuncRef(func_ptr); 
                Option::Some((func_ref, next_compressed_vec, value))
            },
            TypeAction::Passing(arg_type_id) => {
                if (self.get_context().is_vector_type(*arg_type_id)) {
                    let maybe_result = 
                        self.func_opt_state.get_best_vector_to_pass(current_compressed_vec,
                                                    sampled_value_field_state, sampled_embedder_state);
                    let (arg_vec, next_compressed_vec, value) = maybe_result;
                    let arg_ref = TermReference::VecRef(*arg_type_id, to_noisy(arg_vec.view()));
                    Option::Some((arg_ref, next_compressed_vec, value))
                } else {
                    let maybe_result =
                        get_best_term_to_pass(sampled_embedder_state, current_compressed_vec, 
                                                                     sampled_value_field_state);
                    if (maybe_result.is_none()) {
                        return Option::None
                    }
                    let (arg_ptr, next_compressed_vec, value) = maybe_result.unwrap();
                    let arg_ref = TermReference::FuncRef(arg_ptr);
                    Option::Some((arg_ref, next_compressed_vec, value))
                }
            }
        }
    }

    pub fn find_best_next_application(&self, sampled_embedder_state : &SampledEmbedderState,
                                      sampled_value_field_state : &SampledValueFieldState,
                                      current_compressed_vec : &TypedVector) ->
                                      (TermReference, TypedVector, f32) {
        let current_type_id = current_compressed_vec.type_id;
        let target_type_id = self.get_target_type_id();

        let mut best_value = f32::NEG_INFINITY;
        let mut best_term = Option::None;
        let mut best_vec = Option::None;

        let successor_types =  self.type_graph.get_successors(current_type_id);
        for successor_type in successor_types.iter() {
            if self.type_graph.is_reachable_from(*successor_type, target_type_id) {
                let type_actions = self.type_graph.get_actions(current_type_id, *successor_type);
                for type_action in type_actions.iter() {
                    let maybe_next = self.find_best_next_with_transition(sampled_embedder_state,
                                                  sampled_value_field_state,
                                                  current_compressed_vec, type_action);
                    if (!maybe_next.is_none()) {
                        let (term, vec, value) = maybe_next.unwrap();
                        if (value > best_value || best_term.is_none()) {
                            best_term = Option::Some(term.clone());
                            best_vec = Option::Some(vec.clone());
                            best_value = value;
                        }
                    }
                }
            }
        }
        (best_term.unwrap(), best_vec.unwrap(), best_value)
    }

    pub fn find_playout_next_application(&self, sampled_embedder_state : &SampledEmbedderState,
                                         sampled_value_field_state : &SampledValueFieldState,
                                         current_compressed_vec : &TypedVector) ->
                                       (TermReference, TypedVector) {
        let current_type_id = current_compressed_vec.type_id;
        let target_type_id = self.get_target_type_id();

        let successor_types = self.type_graph.get_successors(current_type_id);
    
        let mut rng = rand::thread_rng();

        let mut random_usize : usize = rng.gen();
        let mut successor_ind = random_usize % successor_types.len();
        while (!self.type_graph.is_reachable_from(successor_types[successor_ind], target_type_id)) {
            random_usize = rng.gen();
            successor_ind = random_usize % successor_types.len();
        }

        let successor_type = successor_types[successor_ind];

        let type_actions = self.type_graph.get_actions(current_type_id, successor_type);

        random_usize = rng.gen();
        let action_ind = random_usize % type_actions.len();

        let action = &type_actions[action_ind];

        let (term, vec, _) = self.find_best_next_with_transition(sampled_embedder_state, 
                                                                sampled_value_field_state,
                                                                current_compressed_vec,
                                                                action).unwrap();

        (term, vec)
    }
    
    pub fn find_best_application_chain(&mut self, sampled_embedder_state : &SampledEmbedderState,
                                       sampled_value_field_state : &SampledValueFieldState) -> ApplicationChain {
        let target_type_id = self.get_target_type_id();

        let best_application = self.find_best_application(sampled_embedder_state, sampled_value_field_state);

        let mut current_compressed_vec = sampled_embedder_state.evaluate_term_application(&best_application);
        let mut current_chain = ApplicationChain::from_term_application(best_application);

        let mut current_best_chain = Option::None;
        let mut current_best_value = f32::NEG_INFINITY;

        for _ in 0..OPT_MAX_ITERS {
            //Pick best actions until we get to the target type, then compare with current best
            //chain
            let (picked_term, next_compressed_vec, current_value) = 
                self.find_best_next_application(sampled_embedder_state, sampled_value_field_state,
                                                &current_compressed_vec);

            current_compressed_vec = next_compressed_vec;
            current_chain.add_to_chain(picked_term);

            if (current_compressed_vec.type_id == target_type_id) {
                if (current_value > current_best_value || current_best_chain.is_none()) {
                    current_best_value = current_value;
                    current_best_chain = Option::Some(current_chain.clone());
                } else {
                    //If we've hit the target again, but the chain
                    //has gotten worse, fall back on the current best chain
                    return current_best_chain.unwrap();
                }
            }
        }
        
        while (current_compressed_vec.type_id != target_type_id) { 
            //Pick playout actions until we get to the target type, then yield that
            let (picked_term, next_compressed_vec) = 
                self.find_playout_next_application(sampled_embedder_state, sampled_value_field_state,
                                                   &current_compressed_vec);

            current_compressed_vec = next_compressed_vec;
            current_chain.add_to_chain(picked_term);
        }
        current_chain
    }

    pub fn find_best_application(&mut self, sampled_embedder_state : &SampledEmbedderState,
                                            sampled_value_field_state : &SampledValueFieldState)
                                 -> TermApplication {
        let maybe_nonvec_app = get_best_nonvector_application_with_value(sampled_embedder_state, sampled_value_field_state);
        let maybe_vec_app = self.func_opt_state.update(sampled_embedder_state, sampled_value_field_state);

        if (maybe_nonvec_app.is_none()) {
            let (vec_app, _) = maybe_vec_app.unwrap();
            vec_app
        } else if (maybe_vec_app.is_none()) {
            let (nonvec_app, _) = maybe_nonvec_app.unwrap();
            nonvec_app
        } else {
            let (vec_app, vec_value) = maybe_vec_app.unwrap();
            let (nonvec_app, nonvec_value) = maybe_nonvec_app.unwrap();
            if (vec_value > nonvec_value) {
                vec_app
            } else {
                nonvec_app
            }
        }
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        evaluate_training_data_step(&mut self.interpreter_and_embedder_state, term_ptr, &self.data_points);
    }

    pub fn init_step(&mut self) {
        self.interpreter_and_embedder_state.init_step();
    }

    pub fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>, 
               model_prior_specification : &'a dyn PriorSpecification,
               elaborator_prior_specification : &'a dyn PriorSpecification, ctxt : &'a Context) -> OptimizerState<'a> {

        //Step 1: find the embedding of the target term

        if (data_points.is_empty()) {
            panic!(); 
        }
        info!("Readying interpreter state");
        let mut interpreter_and_embedder_state = InterpreterAndEmbedderState::new(model_prior_specification,
                                                elaborator_prior_specification, ctxt);

        interpreter_and_embedder_state.ensure_every_type_has_a_term_on_init();

        info!("Readying types");

        let func_type_id : TypeId = 2 as TypeId; //Corresponds to scalar function type for us

        info!("Readying target");

        let prior_specification = TermModelPriorSpecification { };
        let mut target_model : TermModel = TermModel::new(func_type_id, &prior_specification, ctxt);

        for (in_vec, out_vec) in data_points.iter() {
            let data_point = DataPoint {
                in_vec : in_vec.clone(),
                out_vec : out_vec.clone(),
                weight : 1.0f32
            };
            target_model.model += data_point;
        }

        let target = target_model.get_schmeared_hole().rescale_spread(TARGET_INV_SCHMEAR_SCALE_FAC);
        let target_type_id = target.type_id;

        let value_field_state = ValueFieldState::new(target, ctxt);

        let func_opt_state = FunctionOptimumState::new(ctxt);

        let type_graph = TypeGraph::build(ctxt);

        OptimizerState {
            interpreter_and_embedder_state,
            value_field_state,
            func_opt_state,
            target_type_id,
            data_points,
            type_graph
        }
    }
}
