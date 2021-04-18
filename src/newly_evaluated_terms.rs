extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use std::collections::HashMap;
use crate::nonprimitive_term_pointer::*;
use crate::type_id::*;
use crate::application_chain::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term_index::*;
use crate::term::*;
use crate::context::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::primitive_term_pointer::*;
use crate::func_impl::*;
use topological_sort::TopologicalSort;

pub struct NewlyEvaluatedTerms {
    pub term_app_results : Vec<TermApplicationResult>,
    pub terms : Vec<NonPrimitiveTermPointer>
}

impl NewlyEvaluatedTerms {
    pub fn new() -> Self {
        NewlyEvaluatedTerms {
            term_app_results : Vec::new(),
            terms : Vec::new()
        }
    }
    pub fn add_term_app_result(&mut self, term_app_result : TermApplicationResult) {
        self.term_app_results.push(term_app_result);
    }
    pub fn add_term(&mut self, term : NonPrimitiveTermPointer) {
        self.terms.push(term);
    }
    pub fn merge(&mut self, mut other : NewlyEvaluatedTerms) {
        for term in other.terms.drain(..) {
            self.terms.push(term);
        }
        for term_app_result in other.term_app_results.drain(..) {
            self.term_app_results.push(term_app_result);
        }
    }
}
