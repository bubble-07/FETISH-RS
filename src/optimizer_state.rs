extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::sampled_embedder_state::*;
use crate::linalg_utils::*;
use crate::application_chain::*;
use crate::type_id::*;
use crate::space_info::*;
use crate::params::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::data_point::*;
use crate::interpreter_and_embedder_state::*;
use crate::model::*;
use crate::inverse_schmear::*;
use crate::schmeared_hole::*;

use crate::value_field_state::*;
use crate::function_optimum_state::*;

extern crate pretty_env_logger;

pub struct OptimizerState {
    pub interpreter_and_embedder_state : InterpreterAndEmbedderState,
    pub value_field_state : ValueFieldState,
    pub func_opt_state : FunctionOptimumState,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>
}

impl OptimizerState {

    pub fn step(&mut self) -> Option<TermPointer> {
        trace!("Applying Bayesian update");
        self.interpreter_and_embedder_state.bayesian_update_step();

        trace!("Sampling embedder state");
        let mut rng = rand::thread_rng();
        let sampled_embedder_state = self.interpreter_and_embedder_state.embedder_state.sample(&mut rng);

        trace!("Updating value fields");
        trace!("Obtaining new constraints");
        let mut constraints = self.interpreter_and_embedder_state.get_new_constraints(&sampled_embedder_state);
        constraints.update_repeat(NUM_CONSTRAINT_REPEATS);
        constraints.update_shuffle();

        trace!("Applying new constraints");
        self.value_field_state.apply_constraints(&constraints);

        trace!("Optimizing for highest-value application chain");
        let best_application_chain = self.find_best_application_chain(&sampled_embedder_state);

        trace!("Done with newly-evaluated terms, discarding");
        self.interpreter_and_embedder_state.clear_newly_received();

        trace!("Evaluating best application");
        let result_ref = self.interpreter_and_embedder_state
                             .interpreter_state.evaluate_application_chain(&best_application_chain);

        if (result_ref.get_type() == self.value_field_state.target.type_id) {
            trace!("Best term was in the target type. Evaluating on training data");
            if let TermReference::FuncRef(func_ptr) = &result_ref {
                self.evaluate_training_data_step(func_ptr.clone());
            }
        }

        match (result_ref) {
            TermReference::FuncRef(func_ptr) => Option::Some(func_ptr),
            TermReference::VecRef(_) => Option::None
        }
    }
    
    pub fn find_best_application_chain(&mut self, 
                                       sampled_embedder_state : &SampledEmbedderState) -> ApplicationChain {
        let best_application = self.find_best_application(sampled_embedder_state); 
        panic!();        
    }

    pub fn find_best_application(&mut self, sampled_embedder_state : &SampledEmbedderState) -> TermApplication {
        let (nonvec_app, nonvec_value) = 
            sampled_embedder_state.get_best_nonvector_application_with_value(&self.interpreter_and_embedder_state.interpreter_state,
                                                                             &self.value_field_state);
        let (vec_app, vec_value) = self.func_opt_state.update(sampled_embedder_state, &self.value_field_state);
        if (vec_value > nonvec_value) {
            vec_app
        } else {
            nonvec_app
        }
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        self.interpreter_and_embedder_state.evaluate_training_data_step(term_ptr, &self.data_points);
    }

    pub fn init_step(&mut self) {
        self.interpreter_and_embedder_state.init_step();
    }

    pub fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>) -> OptimizerState {

        //Step 1: find the embedding of the target term

        if (data_points.is_empty()) {
            panic!(); 
        }
        info!("Readying types");
        let in_dimensions : usize = data_points[0].0.shape()[0];
        let out_dimensions : usize = data_points[0].1.shape()[0];

        let in_type_id : TypeId = get_type_id(&Type::VecType(in_dimensions));
        let out_type_id : TypeId = get_type_id(&Type::VecType(out_dimensions));
        let target_type_id : TypeId = get_type_id(&Type::FuncType(in_type_id, out_type_id));

        info!("Readying interpreter state");
        let interpreter_and_embedder_state = InterpreterAndEmbedderState::new();

        let func_feat_info = get_feature_space_info(target_type_id);

        info!("Readying target");
        
        let mut target_model : Model = Model::new(in_type_id, out_type_id);

        for (in_vec, out_vec) in data_points.iter() {
            let data_point = DataPoint {
                in_vec : in_vec.clone(),
                out_vec : out_vec.clone(),
                weight : 1.0f32
            };
            target_model += data_point;
        }

        let target_inv_schmear : InverseSchmear = target_model.get_inverse_schmear().flatten();

        let normalized_target_inv_schmear = InverseSchmear {
            mean : target_inv_schmear.mean,
            precision : normalize_frob(&target_inv_schmear.precision)
        };

        let sketch_mat = func_feat_info.get_projection_matrix();
        let compressed_target_inv_schmear = normalized_target_inv_schmear.transform_compress(&sketch_mat);

        let target = SchmearedHole {
            type_id : target_type_id,
            full_inv_schmear : normalized_target_inv_schmear,
            compressed_inv_schmear : compressed_target_inv_schmear
        };

        let value_field_state = ValueFieldState::new(target);

        let func_opt_state = FunctionOptimumState::new();

        OptimizerState {
            interpreter_and_embedder_state,
            value_field_state,
            func_opt_state,
            data_points
        }
    }
}
