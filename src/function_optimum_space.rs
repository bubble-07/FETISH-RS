use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;
use std::collections::HashMap;
use crate::model::*;
use crate::normal_inverse_wishart_sampler::*;
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
use crate::data_points::*;

use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

type ModelKey = usize;

pub struct FunctionOptimumSpace {
    pub ret_type : TypeId,
    pub func_space_info : FunctionSpaceInfo,
    pub optimal_input_mapping : Model,
    pub optimal_input_mapping_sampler : NormalInverseWishartSampler,
    pub optimal_input_vectors : HashMap<ModelKey, Array1<f32>>
}

impl FunctionOptimumSpace {
    pub fn estimate_optimal_vector_for_compressed_func(&self, compressed_func : &Array1<f32>) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        let optimal_input_mapping_sample = self.optimal_input_mapping_sampler.sample(&mut rng);

        let func_feats = self.func_space_info.func_feat_info.get_features(compressed_func);
        let optimal_input_vec = optimal_input_mapping_sample.dot(&func_feats);

        optimal_input_vec
    }
    pub fn new(ret_type : TypeId, func_space_info : FunctionSpaceInfo) -> FunctionOptimumSpace {
        let n = func_space_info.in_feat_info.base_dimensions;

        //TODO: Configure parameters for the NIW prior here
        let optimal_input_func_info = func_space_info.build_optimal_input_func_info();

        let optimal_input_mapping = Model::new(optimal_input_func_info);
        let optimal_input_mapping_sampler = NormalInverseWishartSampler::new(&optimal_input_mapping.data);

        let optimal_input_vectors = HashMap::new();

        FunctionOptimumSpace {
            ret_type,
            func_space_info,
            optimal_input_mapping,
            optimal_input_mapping_sampler,
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

        let mut sampled_func_mats : Vec<Array2<f32>> = Vec::new();
        for _ in 0..RANDOM_VECTORS_PER_ITER {
            let mat = self.optimal_input_mapping_sampler.sample(&mut rng);
            sampled_func_mats.push(mat);
        }

        let num_models = sampled_embeddings.models.len();
        let compressed_func_vec_size = self.func_space_info.func_feat_info.get_sketched_dimensions();
        let in_vec_size = self.func_space_info.in_feat_info.get_sketched_dimensions();

        let mut in_model_funcs = Array::zeros((num_models, compressed_func_vec_size));
        let mut out_model_vecs = Array::zeros((num_models, in_vec_size));
        let mut ind = 0;

        for model_key in sampled_embeddings.models.keys() {
            let model_embedding = sampled_embeddings.get_embedding(*model_key);

            let model_compressed_vec = &model_embedding.sampled_compressed_vec;

            //Featurized model embedding matrix
            let model_feat_vec = self.func_space_info.func_feat_info.get_features(model_compressed_vec);

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
            for i in 0..RANDOM_VECTORS_PER_ITER {
                let mat = &sampled_func_mats[i];
                let vec = mat.dot(&model_feat_vec);
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

            in_model_funcs.row_mut(ind).assign(model_compressed_vec);
            out_model_vecs.row_mut(ind).assign(&final_vector);

            self.optimal_input_vectors.insert(*model_key, final_vector);

            ind += 1;
        }

        let data_points = DataPoints {
            in_vecs : in_model_funcs,
            out_vecs : out_model_vecs
        };
        self.optimal_input_mapping += data_points;
        self.optimal_input_mapping_sampler = NormalInverseWishartSampler::new(&self.optimal_input_mapping.data);

        (best_index, best_value)
    }
}
