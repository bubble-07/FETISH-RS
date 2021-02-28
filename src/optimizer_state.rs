extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::trace::*;

use std::collections::HashSet;
use crate::displayable_with_state::*;
use rand::prelude::*;
use std::rc::*;
use crate::constraint_collection::*;
use crate::sampled_embedder_state::*;
use crate::linalg_utils::*;
use crate::array_utils::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::params::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::func_impl::*;
use crate::data_point::*;
use crate::interpreter_and_embedder_state::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;
use crate::schmeared_hole::*;

use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::term_application_result::*;

extern crate pretty_env_logger;

pub struct OptimizerState {
    pub interpreter_and_embedder_state : InterpreterAndEmbedderState,
    target : SchmearedHole,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>
}

impl OptimizerState {

    pub fn step(&mut self) -> Option<TermPointer> {
        let mut rng = rand::thread_rng();

        trace!("Sampling embedder state");
        let sampled_embedder_state = self.interpreter_and_embedder_state.embedder_state.sample(&mut rng);

        trace!("Optimizing for current target");

        trace!("Performing bayesian update");
        self.bayesian_update_step();
        self.interpreter_and_embedder_state.dump_model_traces();

        panic!();
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        self.interpreter_and_embedder_state.evaluate_training_data_step(term_ptr, &self.data_points);
    }

    pub fn bayesian_update_step(&mut self) {
        self.interpreter_and_embedder_state.bayesian_update_step();
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
        
        let target_space = interpreter_and_embedder_state.embedder_state.model_spaces.get(&target_type_id).unwrap();
        let func_space_info = target_space.func_space_info.clone();

        info!("Readying target");
        
        let mut target_model : Model = Model::new(func_space_info);

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

        let sketch_mat = target_space.func_space_info.func_feat_info.get_projection_matrix();
        let compressed_target_inv_schmear = normalized_target_inv_schmear.transform_compress(&sketch_mat);

        let target = SchmearedHole {
            type_id : target_type_id,
            full_inv_schmear : normalized_target_inv_schmear,
            compressed_inv_schmear : compressed_target_inv_schmear
        };

        OptimizerState {
            interpreter_and_embedder_state,
            target,
            data_points
        }
    }
}
