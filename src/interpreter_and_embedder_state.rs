extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::application_chain::*;
use crate::displayable_with_state::*;
use rand::prelude::*;
use crate::context::*;
use crate::vector_application_result::*;
use crate::constraint_collection::*;
use crate::sampled_embedder_state::*;
use crate::linalg_utils::*;
use crate::array_utils::*;
use crate::type_id::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;
use crate::newly_evaluated_terms::*;

use crate::term_application_result::*;

extern crate pretty_env_logger;

pub struct InterpreterAndEmbedderState<'a> {
    pub interpreter_state : InterpreterState<'a>,
    pub embedder_state : EmbedderState<'a>,
    pub newly_evaluated_terms : NewlyEvaluatedTerms
}

impl<'a> InterpreterAndEmbedderState<'a> {
    pub fn get_context(&self) -> &Context {
        self.interpreter_state.get_context()
    }
    pub fn init_step(&mut self) {
    }
    pub fn evaluate(&mut self, term_app : &TermApplication) -> TermReference {
        let (result_ref, newly_evaluated_terms) = self.interpreter_state.evaluate(term_app);
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
        result_ref
    }
    pub fn evaluate_application_chain(&mut self, app_chain : &ApplicationChain) -> TermReference {
        let (result_ref, newly_evaluated_terms) = self.interpreter_state.evaluate_application_chain(app_chain);
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
        result_ref
    }
    pub fn ensure_every_type_has_a_term_on_init(&mut self) {
        let newly_evaluated_terms = self.interpreter_state.ensure_every_type_has_a_term_on_init();
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
    }
    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer, 
                                                  data_points : &Vec<(Array1<f32>, Array1<f32>)>) {
        let mut sq_loss = 0.0f32;

        //Pick a random collection of training points
        for _ in 0..TRAINING_POINTS_PER_ITER {
            let r : usize = rand::thread_rng().gen();
            let i : usize = r % data_points.len();
            let (in_vec, out_vec) = &data_points[i];
            let in_term = TermReference::VecRef(self.get_context().get_arg_type_id(term_ptr.type_id), 
                                                to_noisy(in_vec.view()));
            let term_app = TermApplication {
                func_ptr : term_ptr.clone(),
                arg_ref : in_term
            };
            let (result_ref, newly_evaluated_terms) = self.interpreter_state.evaluate(&term_app);
            if let TermReference::VecRef(_, actual_out_vec_noisy) = result_ref {
                let actual_out_vec = from_noisy(actual_out_vec_noisy.view());
                let loss = sq_vec_dist(out_vec.view(), actual_out_vec.view());
                sq_loss += loss;
            }
            self.newly_evaluated_terms.merge(newly_evaluated_terms);
        }
        info!("Emprirical loss of {} on training set subsample of size {}", sq_loss, TRAINING_POINTS_PER_ITER);
    }
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.bayesian_update_step(&self.interpreter_state, &self.newly_evaluated_terms);
    }
    pub fn clear_newly_received(&mut self) {
        self.newly_evaluated_terms = NewlyEvaluatedTerms::new();
    }

    pub fn new(ctxt : &'a Context) -> InterpreterAndEmbedderState<'a> {
        let interpreter_state = InterpreterState::new(ctxt);
        let embedder_state = EmbedderState::new(ctxt);
        let newly_evaluated_terms = NewlyEvaluatedTerms::new();
        InterpreterAndEmbedderState {
            interpreter_state,
            embedder_state,
            newly_evaluated_terms
        }
    }
}
