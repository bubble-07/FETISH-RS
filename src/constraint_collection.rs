use crate::vector_application_result::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use fetish_lib::everything::*;

pub struct ConstraintCollection<'a> {
    pub constraints : Vec<VectorApplicationResult<'a>>
}

impl<'a> ConstraintCollection<'a> {
    pub fn update_repeat(&mut self, repeats : usize) {
        let mut result = Vec::new();
        for _ in 0..repeats {
            for constraint in &self.constraints {
                result.push(constraint.clone());
            }
        }
        self.constraints = result;
    }
    pub fn update_shuffle(&mut self) {
        let mut rng = thread_rng();
        self.constraints.shuffle(&mut rng);
    }
}

pub fn get_new_constraints<'a>(sampled_embedder_state : &SampledEmbedderState<'a>, 
                           newly_evaluated_terms : &NewlyEvaluatedTerms) -> ConstraintCollection<'a> {
    let mut constraints = Vec::new();
    for term_app_result in &newly_evaluated_terms.term_app_results {
        let func_ptr = term_app_result.term_app.func_ptr;
        let func_type_id = func_ptr.type_id;

        let result_ref = &term_app_result.result_ref;

        //We need to filter out training data evaluations, since they provide no meaningful
        //constraints
        if let TermReference::FuncRef(result_ptr) = result_ref {
            let arg_ref = &term_app_result.term_app.arg_ref;

            let func_vec = sampled_embedder_state.get_model_embedding(func_ptr).sampled_vec.clone();
            let arg_vec = match (arg_ref) {
                TermReference::VecRef(_, vec) => from_noisy(vec.view()),
                TermReference::FuncRef(func_ptr) => 
                    sampled_embedder_state.get_model_embedding(*func_ptr).sampled_vec.clone()
            };
            let ret_vec = sampled_embedder_state.get_model_embedding(*result_ptr).sampled_vec.clone();

            let vector_application_result = VectorApplicationResult {
                func_type_id,
                func_vec,
                arg_vec,
                ret_vec,
                ctxt : sampled_embedder_state.ctxt
            };
            constraints.push(vector_application_result);
        }
    }
    ConstraintCollection {
        constraints
    }
}
