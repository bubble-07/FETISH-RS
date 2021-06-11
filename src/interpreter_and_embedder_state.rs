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
use serde::{Serialize, Deserialize};

///A more convenient, stateful wrapper around an [`InterpreterState`] and [`EmbedderState`]
///which automatically tracks any [`NewlyEvaluatedTerms`] originating from evaluations,
///and an interface which allows for using these to update the wrapped [`EmbedderState`]
///at a caller-chosen time.
pub struct InterpreterAndEmbedderState<'a> {
    pub interpreter_state : InterpreterState<'a>,
    pub embedder_state : EmbedderState<'a>,
    pub newly_evaluated_terms : NewlyEvaluatedTerms
}

#[derive(Serialize, Deserialize)]
pub struct SerializedInterpreterAndEmbedderState {
    pub interpreter_state : SerializedInterpreterState,
    pub embedder_state : SerializedEmbedderState,
    pub newly_evaluated_terms : NewlyEvaluatedTerms
}

impl SerializedInterpreterAndEmbedderState {
    pub fn deserialize<'a>(self, ctxt : &'a Context) -> InterpreterAndEmbedderState<'a> {
        InterpreterAndEmbedderState {
            interpreter_state : self.interpreter_state.deserialize(ctxt),
            embedder_state : self.embedder_state.deserialize(ctxt),
            newly_evaluated_terms : self.newly_evaluated_terms
        }
    }
}

impl<'a> InterpreterAndEmbedderState<'a> {
    pub fn serialize(self) -> SerializedInterpreterAndEmbedderState {
        SerializedInterpreterAndEmbedderState {
            interpreter_state : self.interpreter_state.serialize(),
            embedder_state : self.embedder_state.serialize(),
            newly_evaluated_terms : self.newly_evaluated_terms
        }
    }

    ///Gets the [`Context`] that this [`InterpreterAndEmbedderState`] exists in.
    pub fn get_context(&self) -> &Context {
        self.interpreter_state.get_context()
    }
    ///Given a [`TermApplication`], uses the wrapped [`InterpreterState`] to evaluate
    ///the application, returning a `TermReference` for the result of the evaluation.
    ///Any newly-evaluated terms which result from evaluation will be added to
    ///this [`InterpreterAndEmbedderState`]'s `newly_evaluated_terms` member variable.
    pub fn evaluate(&mut self, term_app : &TermApplication) -> TermReference {
        let (result_ref, newly_evaluated_terms) = self.interpreter_state.evaluate(term_app);
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
        result_ref
    }
    ///Convenience method to force the wrapped [`InterpreterState`] to have at least
    ///one term inhabiting every type, assuming that it doesn't really matter what these are.
    ///Calling this method will result in every newly-added term being added to the
    ///wrapped [`NewlyEvaluatedTerms`]
    pub fn ensure_every_type_has_a_term_on_init(&mut self) {
        let newly_evaluated_terms = self.interpreter_state.ensure_every_type_has_a_term_on_init();
        self.newly_evaluated_terms.merge(newly_evaluated_terms);
    }
    ///Uses the wrapped [`NewlyEvaluatedTerms`] and [`InterpreterState`] to update the embeddings
    ///within the wrapped [`EmbedderState`]. Calling this method will not modfiy the wrapped
    ///[`NewlyEvaluatedTerms`], in case they are still of use after an embedding update in
    ///your particular use-case.
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.bayesian_update_step(&self.interpreter_state, &self.newly_evaluated_terms);
    }

    ///Clears out the wrapped [`NewlyEvaluatedTerms`], which typically will indicate that 
    ///a new cycle of evaluations of terms against the [`InterpreterState`] is about to begin.
    pub fn clear_newly_received(&mut self) {
        self.newly_evaluated_terms = NewlyEvaluatedTerms::new();
    }

    ///Constructs a new [`InterpreterAndEmbedderState`] with the given [`Context`].
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
