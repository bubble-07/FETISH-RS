use crate::application_chain::*;
use fetish_lib::everything::*;

pub fn interpreter_evaluate_application_chain(state : &mut InterpreterState, app_chain : &ApplicationChain) -> (TermReference, NewlyEvaluatedTerms) {
    let mut newly_evaluated_terms = NewlyEvaluatedTerms::new();
    let mut current_ref = app_chain.term_refs[0].clone();
    for i in 1..app_chain.term_refs.len() {
        let current_type = current_ref.get_type();

        let other_ref = &app_chain.term_refs[i];
        let other_type = other_ref.get_type();

        let mut current_is_applicative = false;
        if (!state.ctxt.is_vector_type(current_type)) {
            let arg_type = state.ctxt.get_arg_type_id(current_type);
            current_is_applicative = (arg_type == other_type);
        }

        let term_app = if (current_is_applicative) {
            match (current_ref) {
                TermReference::FuncRef(current_ptr) => {
                    TermApplication {
                        func_ptr : current_ptr,
                        arg_ref : other_ref.clone()
                    }
                },
                TermReference::VecRef(_, _) => { panic!(); }
            }
        } else {
            match (other_ref) {
                TermReference::FuncRef(other_ptr) => {
                    TermApplication {
                        func_ptr : other_ptr.clone(),
                        arg_ref : current_ref
                    }
                },
                TermReference::VecRef(_, _) => { panic!(); }
            }
        };

        let eval_pair = state.evaluate(&term_app);
        current_ref = eval_pair.0;
        newly_evaluated_terms.merge(eval_pair.1);
    }
    (current_ref, newly_evaluated_terms)
}

