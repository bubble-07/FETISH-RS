extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::displayable_with_state::*;
use crate::prior_specification::*;
use rand::prelude::*;
use crate::context::*;
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
    pub fn ensure_every_type_has_a_term_on_init(&mut self) {
        let newly_evaluated_terms = self.interpreter_state.ensure_every_type_has_a_term_on_init();
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
    }
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.bayesian_update_step(&self.interpreter_state, &self.newly_evaluated_terms);
    }
    pub fn clear_newly_received(&mut self) {
        self.newly_evaluated_terms = NewlyEvaluatedTerms::new();
    }

    pub fn new(model_prior_specification : &'a dyn PriorSpecification,
               elaborator_prior_specification : &'a dyn PriorSpecification, 
               ctxt : &'a Context) -> InterpreterAndEmbedderState<'a> {
        let interpreter_state = InterpreterState::new(ctxt);
        let embedder_state = EmbedderState::new(model_prior_specification, elaborator_prior_specification, ctxt);
        let newly_evaluated_terms = NewlyEvaluatedTerms::new();
        InterpreterAndEmbedderState {
            interpreter_state,
            embedder_state,
            newly_evaluated_terms
        }
    }
}
