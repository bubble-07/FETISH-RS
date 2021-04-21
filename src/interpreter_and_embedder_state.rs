use ndarray::*;
use fetish_lib::everything::*;
use crate::application_chain::*;
use crate::params::*;
use crate::interpreter_state::*;
use rand::prelude::*;

pub fn evaluate_application_chain(state : &mut InterpreterAndEmbedderState, app_chain : &ApplicationChain) -> TermReference {
    let (result_ref, newly_evaluated_terms) = interpreter_evaluate_application_chain(&mut state.interpreter_state, app_chain);
    state.newly_evaluated_terms.merge(newly_evaluated_terms);
    result_ref
}
pub fn evaluate_training_data_step(state : &mut InterpreterAndEmbedderState, term_ptr : TermPointer, 
                                              data_points : &Vec<(Array1<f32>, Array1<f32>)>) {
    let mut sq_loss = 0.0f32;

    //Pick a random collection of training points
    for _ in 0..TRAINING_POINTS_PER_ITER {
        let r : usize = rand::thread_rng().gen();
        let i : usize = r % data_points.len();
        let (in_vec, out_vec) = &data_points[i];
        let in_term = TermReference::VecRef(state.get_context().get_arg_type_id(term_ptr.type_id), 
                                            to_noisy(in_vec.view()));
        let term_app = TermApplication {
            func_ptr : term_ptr.clone(),
            arg_ref : in_term
        };
        let (result_ref, newly_evaluated_terms) = state.interpreter_state.evaluate(&term_app);
        if let TermReference::VecRef(_, actual_out_vec_noisy) = result_ref {
            let actual_out_vec = from_noisy(actual_out_vec_noisy.view());
            let loss = sq_vec_dist(out_vec.view(), actual_out_vec.view());
            sq_loss += loss;
        }
        state.newly_evaluated_terms.merge(newly_evaluated_terms);
    }
    info!("Emprirical loss of {} on training set subsample of size {}", sq_loss, TRAINING_POINTS_PER_ITER);
}
