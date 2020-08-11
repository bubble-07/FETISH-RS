extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::collections::HashSet;
use crate::displayable_with_state::*;
use rand::prelude::*;
use std::rc::*;
use crate::linalg_utils::*;
use crate::array_utils::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::params::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::func_impl::*;
use crate::bayes_utils::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;
use either::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::term_application_result::*;

extern crate pretty_env_logger;

pub struct OptimizerStateWithTarget {
    pub optimizer_state : OptimizerState,
    target_inv_schmear : InverseSchmear,
    target_type_id : TypeId,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>
}

pub struct OptimizerState {
    pub interpreter_state : InterpreterState,
    embedder_state : EmbedderState,
}

impl OptimizerStateWithTarget {
    pub fn step(&mut self) -> TermPointer {
        let result : TermPointer = self.optimize_evaluate_step();
        self.evaluate_training_data_step(result.clone());
        self.bayesian_update_step();
        result
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        self.optimizer_state.evaluate_training_data_step(term_ptr, &self.data_points);
    }

    pub fn optimize_evaluate_step(&mut self) -> TermPointer {
        self.optimizer_state.optimize_evaluate_step(self.target_type_id, &self.target_inv_schmear)
    }
    pub fn bayesian_update_step(&mut self) {
        self.optimizer_state.bayesian_update_step();
    }
    pub fn init_step(&mut self) {
        self.optimizer_state.init_step();
    }

    pub fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>) -> OptimizerStateWithTarget {

        //Step 1: find the embedding of the target term

        if (data_points.is_empty()) {
            panic!(); 
        }
        info!("Readying types");
        let in_dimensions : usize = data_points[0].0.shape()[0];
        let out_dimensions : usize = data_points[0].1.shape()[0];

        let in_type_id : TypeId = get_type_id(&Type::VecType(in_dimensions));
        let out_type_id : TypeId = get_type_id(&Type::VecType(out_dimensions));
        let target_type_id : TypeId = get_type_id(&Type::FuncType(in_type_id, out_type_id));

        info!("Readying interpreter state");
        let optimizer_state = OptimizerState::new();
        
        let target_space = optimizer_state.embedder_state.model_spaces.get(&target_type_id).unwrap();
        let rc_feature_collections = target_space.feature_collections.clone();

        info!("Readying target");
        
        let mut target_model : Model = Model::new(rc_feature_collections, in_dimensions, out_dimensions);

        let out_precision = Array::eye(out_dimensions);

        for (in_vec, out_vec) in data_points.iter() {
            let data_point = DataPoint {
                in_vec : in_vec.clone(),
                out_inv_schmear : InverseSchmear {
                    mean : out_vec.clone(),
                    precision : out_precision.clone()
                }
            };
            target_model += data_point;
        }

        let target_inv_schmear : InverseSchmear = target_model.get_inverse_schmear().flatten();

        let target_space = optimizer_state.embedder_state.model_spaces.get(&target_type_id).unwrap();
        let reduced_target_inv_schmear = target_space.compress_inverse_schmear(&target_inv_schmear);

        OptimizerStateWithTarget {
            optimizer_state,
            target_inv_schmear : reduced_target_inv_schmear,
            target_type_id,
            data_points
        }
    }
}

impl OptimizerState {
    pub fn init_step(&mut self) {
        self.embedder_state.init_embeddings(&mut self.interpreter_state);
    }
    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer, 
                                                  data_points : &Vec<(Array1<f32>, Array1<f32>)>) {
        let mut sq_loss = 0.0f32;

        //Pick a random collection of training points
        for _ in 0..TRAINING_POINTS_PER_ITER {
            let r : usize = rand::thread_rng().gen();
            let i : usize = r % data_points.len();
            let (in_vec, out_vec) = &data_points[i];
            let in_term = TermReference::from(in_vec);
            let term_app = TermApplication {
                func_ptr : term_ptr.clone(),
                arg_ref : in_term
            };
            let result_ref = self.interpreter_state.evaluate(&term_app);
            if let TermReference::VecRef(actual_out_vec_noisy) = result_ref {
                let actual_out_vec = from_noisy(&actual_out_vec_noisy);
                let loss = sq_vec_dist(out_vec, &actual_out_vec);
                sq_loss += loss;
            }
        }
        info!("Emprirical loss of {} on training set subsample of size {}", sq_loss, TRAINING_POINTS_PER_ITER);
    }
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.init_embeddings(&mut self.interpreter_state);
        let mut data_updated_terms : HashSet<TermPointer> = HashSet::new();
        let mut prior_updated_terms : HashSet<TermPointer> = HashSet::new();

        let mut updated_apps : HashSet::<TermApplicationResult> = HashSet::new();
        for term_app_result in self.interpreter_state.new_term_app_results.drain(..) {
            updated_apps.insert(term_app_result); 
        }

        trace!("Propagating data updates for {} applications", updated_apps.len());
        self.embedder_state.propagate_data_recursive(&self.interpreter_state, updated_apps, &mut data_updated_terms);
        trace!("Propagating prior updates for {} applications", data_updated_terms.len());
        self.embedder_state.propagate_prior_recursive(&self.interpreter_state, data_updated_terms, &mut prior_updated_terms);
    }

    fn optimize_evaluate_step(&mut self, target_type_id : TypeId, target_inv_schmear : &InverseSchmear)
                                     -> TermPointer {
        //Sample for the best term in the space of the target 
        trace!("Thompson sampling for term of type {}", get_type(target_type_id));
        let (term_pointer, term_dist) = self.embedder_state.thompson_sample_term(target_type_id, target_inv_schmear);
        trace!("Obtained {} with loss {}", term_pointer.display(&self.interpreter_state), term_dist);

        //Now, iterate through all term applications yielding the type of the target
        let mut application_type_ids : Vec::<(TypeId, TypeId)> = get_application_type_ids(target_type_id);
        let mut best_dist : f32 = f32::INFINITY;
        let mut best_application_and_types : Option<(TermApplication, TypeId, TypeId)> = Option::None;
        
        for (func_type_id, arg_type_id) in application_type_ids.drain(..) {
            trace!("Thompson sampling for application of {}", get_type(func_type_id));
            let func_space = self.embedder_state.model_spaces.get(&func_type_id).unwrap();
            let (application, dist) = self.embedder_state.thompson_sample_app(func_type_id, arg_type_id, 
                                                                              target_inv_schmear);

            trace!("Obtained {} with loss {}", application.display(&self.interpreter_state), dist);
            if (dist < best_dist) {
                best_application_and_types = Option::Some((application, func_type_id, arg_type_id));
                best_dist = dist;
            }
        }
        match best_application_and_types {
            Option::None => term_pointer,
            Option::Some(application_and_types) => {
                let (application, func_type_id, arg_type_id) = application_and_types;
                if (best_dist < term_dist) {
                    let target_mean = &target_inv_schmear.mean;
                    let (func_target, maybe_arg_target) = 
                        self.embedder_state.find_better_app(&application, target_mean);

                    info!("Recursing on function [");
                    let better_func = self.optimize_evaluate_step(func_type_id, &func_target);
                    info!("]");

                    let better_arg : TermReference = match (maybe_arg_target) {
                        Left(arg_target) => {
                            info!("Recursing on arg [");
                            let better_arg = self.optimize_evaluate_step(arg_type_id, &arg_target);
                            info!("]");
                            TermReference::FuncRef(better_arg)
                        },
                        Right(arg_vec) => {
                            //In this case, we optimized the function choice and the argument
                            //choice, but the argument was a vector, so we need to store it
                            self.embedder_state.store_vec(arg_type_id, arg_vec.clone());
                            TermReference::VecRef(arg_vec)
                        }
                    };
                    let better_application = TermApplication {
                        func_ptr : better_func,
                        arg_ref : better_arg
                    };
                    let result_ref = self.interpreter_state.evaluate(&better_application);
                    if let TermReference::FuncRef(ret_ptr) = result_ref {
                        ret_ptr
                    } else {
                        //This should not happen, because we should never be using the optimizer
                        //to attempt to find applications yielding bare vectors
                        panic!();
                    }
                } else {
                    term_pointer
                }
            }
        }

    }
    fn new() -> OptimizerState {
        OptimizerState {
            interpreter_state : InterpreterState::new(),
            embedder_state : EmbedderState::new()
        }
    }
}
