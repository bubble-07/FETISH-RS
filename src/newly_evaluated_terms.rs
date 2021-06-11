use crate::nonprimitive_term_pointer::*;
use crate::type_id::*;
use crate::term::*;
use crate::term_application_result::*;
use crate::primitive_term_pointer::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

///A collection of [`TermApplicationResult`]s and [`NonPrimitiveTermPointer`]s
///which were generated as a consequence of new evaluations performed
///by an [`crate::interpreter_state::InterpreterState`].
#[derive(Serialize, Deserialize)]
pub struct NewlyEvaluatedTerms {
    pub term_app_results : Vec<TermApplicationResult>,
    pub terms : Vec<NonPrimitiveTermPointer>
}

impl NewlyEvaluatedTerms {
    ///Yields an initially-empty [`NewlyEvaluatedTerms`].
    pub fn new() -> Self {
        NewlyEvaluatedTerms {
            term_app_results : Vec::new(),
            terms : Vec::new()
        }
    }

    ///Yields a [`HashMap`] mapping from [`TermApplicationResult`]s to the
    ///count at which each occurs within this [`NewlyEvaluatedTerms`].
    pub fn get_count_map(&self) -> HashMap<TermApplicationResult, usize> {
        let mut result = HashMap::new();
        for term_app_result in self.term_app_results.iter() {
            let the_clone = term_app_result.clone();
            if (!result.contains_key(term_app_result)) {
                result.insert(the_clone, 1);
            } else {
                let prev_count = *result.get(term_app_result).unwrap();
                result.insert(the_clone, prev_count + 1);
            }
        }
        result
    }

    ///Adds the given [`TermApplicationResult`] to the list of new term applications.
    pub fn add_term_app_result(&mut self, term_app_result : TermApplicationResult) {
        self.term_app_results.push(term_app_result);
    }
    ///Adds the given [`NonPrimitiveTermPointer`] to the list of new terms.
    pub fn add_term(&mut self, term : NonPrimitiveTermPointer) {
        self.terms.push(term);
    }
    ///Merges in the new terms and term application results from the passed in
    ///other [`NewlyEvaluatedTerms`].
    pub fn merge(&mut self, mut other : NewlyEvaluatedTerms) {
        for term in other.terms.drain(..) {
            self.terms.push(term);
        }
        for term_app_result in other.term_app_results.drain(..) {
            self.term_app_results.push(term_app_result);
        }
    }
}
