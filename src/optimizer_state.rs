extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::collections::HashSet;
use crate::displayable_with_state::*;
use rand::prelude::*;
use std::rc::*;
use crate::sampled_embedder_state::*;
use crate::linalg_utils::*;
use crate::array_utils::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::params::*;
use crate::schmeared_hole::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::func_impl::*;
use crate::data_point::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;
use crate::featurization_inverse_directory::*;
use crate::bounded_hole::*;
use crate::linear_expression_queue::*;

use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::term_application_result::*;

extern crate pretty_env_logger;

pub struct OptimizerStateWithTarget {
    pub optimizer_state : OptimizerState,
    target : SchmearedHole,
    data_points : Vec::<(Array1<f32>, Array1<f32>)>
}

pub struct OptimizerState {
    pub interpreter_state : InterpreterState,
    embedder_state : EmbedderState,
    feat_inverse_directory : FeaturizationInverseDirectory
}

impl OptimizerStateWithTarget {
    fn get_current_target_hole(&self, embedder_state : &SampledEmbedderState) -> BoundedHole {
        self.target.get_closer_than_closest_term_bound(embedder_state)
    }

    pub fn step(&mut self) -> TermPointer {
        let mut rng = rand::thread_rng();

        trace!("Sampling embedder state");
        let sampled_embedder_state = self.optimizer_state.embedder_state.sample(&mut rng);
        let mut lin_expr_queue = LinearExpressionQueue::new();

        trace!("Finding current target");
        let target_hole = self.get_current_target_hole(&sampled_embedder_state);

        trace!("Optimizing for current target");
        let (lin_expr, feat_points_directory) = lin_expr_queue.find_within_bound(&target_hole, 
                                                &self.optimizer_state.interpreter_state,
                                                &self.optimizer_state.embedder_state,
                                                &sampled_embedder_state, 
                                                &self.optimizer_state.feat_inverse_directory);

        trace!("Updating featurization inverse directory");
        //Update the featurization inverse directory
        self.optimizer_state.feat_inverse_directory += feat_points_directory;

        trace!("Evaluating found term");

        //Evaluate the expression that we obtained
        let term_ref = self.optimizer_state.interpreter_state.evaluate_linear_expression(lin_expr);
        info!("Current best term: {}", term_ref.display(&self.optimizer_state.interpreter_state));
        if let TermReference::FuncRef(result) = term_ref {
            trace!("Evaluating on training data");
            self.evaluate_training_data_step(result.clone());
            trace!("Performing bayesian update");
            self.bayesian_update_step();
            return result.clone();
        }
        error!("Wrong type for the linear expression");
        panic!();
    }

    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer) {
        self.optimizer_state.evaluate_training_data_step(term_ptr, &self.data_points);
    }

    pub fn bayesian_update_step(&mut self) {
        self.optimizer_state.bayesian_update_step();
    }
    pub fn init_step(&mut self) {
        self.optimizer_state.init_step();
    }

    pub fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>) -> OptimizerStateWithTarget {

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
        let optimizer_state = OptimizerState::new();
        
        let target_space = optimizer_state.embedder_state.model_spaces.get(&target_type_id).unwrap();
        let space_info = target_space.space_info.clone();

        info!("Readying target");
        
        let mut target_model : Model = Model::new(space_info);

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

        let target = SchmearedHole {
            type_id : target_type_id,
            inv_schmear : normalized_target_inv_schmear
        };

        OptimizerStateWithTarget {
            optimizer_state,
            target,
            data_points
        }
    }
}

impl OptimizerState {
    pub fn init_step(&mut self) {
        self.embedder_state.init_embeddings(&mut self.interpreter_state);
        self.bayesian_update_step();
    }
    pub fn evaluate_training_data_step(&mut self, term_ptr : TermPointer, 
                                                  data_points : &Vec<(Array1<f32>, Array1<f32>)>) {
        let mut sq_loss = 0.0f32;

        //Pick a random collection of training points
        for _ in 0..TRAINING_POINTS_PER_ITER {
            let r : usize = rand::thread_rng().gen();
            let i : usize = r % data_points.len();
            let (in_vec, out_vec) = &data_points[i];
            let in_term = TermReference::from(in_vec);
            let term_app = TermApplication {
                func_ptr : term_ptr.clone(),
                arg_ref : in_term
            };
            let result_ref = self.interpreter_state.evaluate(&term_app);
            if let TermReference::VecRef(actual_out_vec_noisy) = result_ref {
                let actual_out_vec = from_noisy(&actual_out_vec_noisy);
                let loss = sq_vec_dist(out_vec, &actual_out_vec);
                sq_loss += loss;
            }
        }
        info!("Emprirical loss of {} on training set subsample of size {}", sq_loss, TRAINING_POINTS_PER_ITER);
    }
    pub fn bayesian_update_step(&mut self) {
        self.embedder_state.init_embeddings(&mut self.interpreter_state);
        let mut data_updated_terms : HashSet<TermPointer> = HashSet::new();
        let mut prior_updated_terms : HashSet<TermPointer> = HashSet::new();

        let mut updated_apps : HashSet::<TermApplicationResult> = HashSet::new();
        for term_app_result in self.interpreter_state.new_term_app_results.drain(..) {
            updated_apps.insert(term_app_result); 
        }

        trace!("Propagating data updates for {} applications", updated_apps.len());
        self.embedder_state.propagate_data_recursive(&self.interpreter_state, updated_apps, &mut data_updated_terms);
        trace!("Propagating prior updates for {} applications", data_updated_terms.len());
        self.embedder_state.propagate_prior_recursive(&self.interpreter_state, data_updated_terms, &mut prior_updated_terms);
    }

    fn new() -> OptimizerState {
        let embedder_state = EmbedderState::new();
        let feat_inverse_directory = FeaturizationInverseDirectory::new(&embedder_state);
        OptimizerState {
            interpreter_state : InterpreterState::new(),
            embedder_state,
            feat_inverse_directory
        }
    }
}
