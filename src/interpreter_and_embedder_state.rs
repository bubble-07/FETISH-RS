extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::trace::*;

use std::collections::HashSet;
use crate::displayable_with_state::*;
use rand::prelude::*;
use std::rc::*;
use crate::vector_application_result::*;
use crate::constraint_collection::*;
use crate::sampled_embedder_state::*;
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
use crate::data_point::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;
use crate::schmeared_hole::*;

use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::term_application_result::*;

extern crate pretty_env_logger;

pub struct InterpreterAndEmbedderState {
    pub interpreter_state : InterpreterState,
    pub embedder_state : EmbedderState
}

impl InterpreterAndEmbedderState {
    pub fn dump_model_traces(&self) {
        for (type_id, model_space) in self.embedder_state.model_spaces.iter() {
            let resolved_type = get_type(*type_id);
            info!("{}:", resolved_type);
            for (term_index, model) in model_space.models.iter() {
                let term_pointer = TermPointer {
                    type_id : *type_id,
                    index : *term_index
                };
                let term_string = term_pointer.display(&self.interpreter_state);
                let distr = &model.model.data;
                let trace = distr.sigma.trace().unwrap();
                info!("{} : {}", term_string, trace);
            }
            info!("\n");
        }
    }
    pub fn init_step(&mut self) {
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
    pub fn get_new_constraints(&self, sampled_embedder_state : &SampledEmbedderState) -> ConstraintCollection {
        let mut constraints = Vec::new();
        for term_app_result in &self.interpreter_state.new_term_app_results {
            let func_ptr = &term_app_result.term_app.func_ptr;
            let func_type_id = func_ptr.type_id;

            let ret_type_id = get_ret_type_id(func_type_id);

            //We need to filter out training data evaluations, since they provide no meaningful
            //constraints
            if (!is_vector_type(ret_type_id)) {
                let arg_ref = &term_app_result.term_app.arg_ref;
                let result_ref = &term_app_result.result_ref;

                let func_vec = sampled_embedder_state.get_model_embedding(func_ptr).sampled_vec.clone();
                let arg_vec = sampled_embedder_state.get_term_embedding(arg_ref).get_flattened();
                let ret_vec = sampled_embedder_state.get_term_embedding(result_ref).get_flattened();

                let vector_application_result = VectorApplicationResult {
                    func_type_id,
                    func_vec,
                    arg_vec,
                    ret_vec
                };
                constraints.push(vector_application_result);
            }
        }
        ConstraintCollection {
            constraints
        }
    }
    pub fn get_new_term_app_results(&self) -> &Vec<TermApplicationResult> {
        &self.interpreter_state.new_term_app_results
    }
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.bayesian_update_step(&self.interpreter_state);
    }
    pub fn clear_newly_received(&mut self) {
        self.interpreter_state.clear_newly_received();
    }

    pub fn new() -> InterpreterAndEmbedderState {
        let embedder_state = EmbedderState::new();
        InterpreterAndEmbedderState {
            interpreter_state : InterpreterState::new(),
            embedder_state
        }
    }
}
