use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;
use std::collections::HashMap;
use crate::params::*;
use crate::sampled_term_embedding::*;
use crate::sampled_model_embedding::*;
use crate::function_space_info::*;
use crate::value_field_maximum_solver::*;
use crate::schmear::*;
use crate::schmear_sampler::*;
use crate::function_space_directory::*;
use crate::type_id::*;
use crate::sampled_embedder_state::*;
use crate::sampled_embedding_space::*;
use crate::value_field_state::*;

use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

type ModelKey = usize;

pub struct FunctionOptimumSpace {
    pub ret_type : TypeId,
    pub func_space_info : FunctionSpaceInfo,
    pub optimal_input_schmear : Schmear,
    pub optimal_input_schmear_sampler : SchmearSampler,
    pub optimal_input_vectors : HashMap<ModelKey, Array1<f32>>
}

impl FunctionOptimumSpace {
    pub fn new(ret_type : TypeId, func_space_info : FunctionSpaceInfo) -> FunctionOptimumSpace {
        let n = func_space_info.in_feat_info.base_dimensions;
        let mean = Array::zeros((n,));
        let mut covariance = Array::eye(n);
        covariance *= INITIAL_FUNCTION_OPTIMUM_VARIANCE;
        let optimal_input_schmear = Schmear {
            mean,
            covariance
        };
        let optimal_input_schmear_sampler = SchmearSampler::new(&optimal_input_schmear);

        let optimal_input_vectors = HashMap::new();

        FunctionOptimumSpace {
            ret_type,
            func_space_info,
            optimal_input_schmear,
            optimal_input_schmear_sampler,
            optimal_input_vectors
        }
    }
    pub fn get_optimal_vector(&self, model_key : ModelKey) -> &Array1<f32> {
        self.optimal_input_vectors.get(&model_key).unwrap()
    }
    pub fn update(&mut self, sampled_embeddings : &SampledEmbeddingSpace, 
                             value_field_state : &ValueFieldState) -> (usize, f32) {
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;

        let mut rng = rand::thread_rng();
        let maybe_target_schmear = value_field_state.get_target_for_type(self.ret_type);
        let target_compressed_inv_schmear = match (maybe_target_schmear) {
                                                Option::None => Option::None,
                                                Option::Some(hole) => Option::Some(hole.compressed_inv_schmear)
                                            };
        let value_field_coefs = &value_field_state.get_value_field(self.ret_type).coefs;

        let mut optimized_vectors : Vec<Array1<f32>> = Vec::new();

        for model_key in sampled_embeddings.models.keys() {
            let model_embedding = sampled_embeddings.get_embedding(*model_key);

            let value_field_max_solver = ValueFieldMaximumSolver {
                func_space_info : self.func_space_info.clone(),
                func_mat : model_embedding.sampled_mat.clone(),
                value_field_coefs : value_field_coefs.clone(),
                target_compressed_inv_schmear : target_compressed_inv_schmear.clone()
            };

            let mut possible_initial_vectors : Vec<Array1<f32>> = Vec::new();
            if (self.optimal_input_vectors.contains_key(model_key)) {
                let vec = self.optimal_input_vectors.get(model_key).unwrap();
                possible_initial_vectors.push(vec.clone());
            }
            for _ in 0..RANDOM_VECTORS_PER_ITER {
                let vec = self.optimal_input_schmear_sampler.sample(&mut rng);
                possible_initial_vectors.push(vec);
            }

            let initial_vector = value_field_max_solver.get_compressed_vector_with_max_value(&possible_initial_vectors);
            
            //Optimize starting from the inital vector to yield a new optimal point
            let linesearch = MoreThuenteLineSearch::new();
            let solver = SteepestDescent::new(linesearch);

            let maybe_result = Executor::new(value_field_max_solver, solver, initial_vector.clone())
                                        .max_iters(NUM_STEEPEST_DESCENT_STEPS_PER_ITER as u64)
                                        .run();

            let final_vector = match (maybe_result) {
                Ok(result) => {
                    if (-result.state.cost > best_value) {
                        best_value = -result.state.cost;
                        best_index = *model_key;
                    }
                    result.state.param
                },
                Err(e) => {
                    error!("Unexpected error on optimization: {}", e);
                    initial_vector.clone()
                }
            };

            optimized_vectors.push(final_vector.clone());
            self.optimal_input_vectors.insert(*model_key, final_vector);
        }
        let empirical_optimal_input_schmear = Schmear::from_sample_vectors(&optimized_vectors);
        self.optimal_input_schmear.update_lerp(empirical_optimal_input_schmear, LERP_FACTOR);
        self.optimal_input_schmear_sampler = SchmearSampler::new(&self.optimal_input_schmear);
        (best_index, best_value)
    }
    fn add_vector(&mut self, model_key : ModelKey) {
        let mut rng = rand::thread_rng();
        let vec = self.optimal_input_schmear_sampler.sample(&mut rng);
        self.optimal_input_vectors.insert(model_key, vec);
    }
}
