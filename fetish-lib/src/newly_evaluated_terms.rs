use crate::nonprimitive_term_pointer::*;
use crate::type_id::*;
use crate::term::*;
use crate::term_application_result::*;
use crate::primitive_term_pointer::*;

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
