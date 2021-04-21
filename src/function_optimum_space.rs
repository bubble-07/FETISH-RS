use ndarray::*;
use std::collections::HashMap;
use fetish_lib::everything::*;
use crate::value_field_maximum_solver::*;
use crate::sampled_value_field_state::*;
use crate::params::*;

use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

type ModelKey = TermIndex;

pub struct FunctionOptimumSpace<'a> {
    pub func_type_id : TypeId,
    pub optimal_input_mapping : Model<'a>,
    pub optimal_input_mapping_sampler : NormalInverseWishartSampler,
    pub optimal_input_vectors : HashMap<ModelKey, Array1<f32>>,
    pub ctxt : &'a Context
}

struct FunctionOptimumPriorSpecification {
}

impl PriorSpecification for FunctionOptimumPriorSpecification {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        FUNC_OPTIMUM_IN_PRECISION_MULTIPLIER
    }
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32 {
        let pseudo_observations = self.get_out_pseudo_observations(out_dims);
        pseudo_observations * FUNC_OPTIMUM_OUT_COVARIANCE_MULTIPLIER
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        //we need to ensure that the model and its covariance are always well-specified
        (out_dims as f32) * FUNC_OPTIMUM_ERROR_COVARIANCE_PRIOR_OBSERVATIONS_PER_DIMENSION + 4.0f32
    }
}

impl <'a> FunctionOptimumSpace<'a> {
    pub fn estimate_optimal_vector_for_compressed_func(&self, compressed_func : ArrayView1<f32>) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        let optimal_input_mapping_sample = self.optimal_input_mapping_sampler.sample(&mut rng);

        let func_feat_info = self.ctxt.get_feature_space_info(self.func_type_id);
        let func_feats = func_feat_info.get_features(compressed_func);
        let optimal_input_vec = optimal_input_mapping_sample.dot(&func_feats);

        optimal_input_vec
    }

    pub fn new(func_type_id : TypeId, ctxt : &'a Context) -> FunctionOptimumSpace {

        let prior_specification = FunctionOptimumPriorSpecification { };

        let optimal_input_mapping = Model::new(&prior_specification, func_type_id, ctxt.get_arg_type_id(func_type_id), ctxt);
        let optimal_input_mapping_sampler = NormalInverseWishartSampler::new(&optimal_input_mapping.data);

        let optimal_input_vectors = HashMap::new();

        FunctionOptimumSpace {
            func_type_id,
            optimal_input_mapping,
            optimal_input_mapping_sampler,
            optimal_input_vectors,
            ctxt
        }
    }
    pub fn get_optimal_vector(&self, model_key : ModelKey) -> &Array1<f32> {
        self.optimal_input_vectors.get(&model_key).unwrap()
    }
    pub fn update(&mut self, sampled_embeddings : &SampledEmbeddingSpace, 
                             value_field_state : &SampledValueFieldState) -> Option<(TermIndex, f32)> {
        let mut best_index = Option::None;
        let mut best_value = f32::NEG_INFINITY;

        let func_space_info = self.ctxt.get_function_space_info(self.func_type_id);
        let func_feat_info = self.ctxt.get_feature_space_info(self.func_type_id);
        let ret_type = self.ctxt.get_ret_type_id(self.func_type_id);

        let mut rng = rand::thread_rng();
        let value_field = value_field_state.get_value_field(ret_type);

        let mut sampled_func_mats : Vec<Array2<f32>> = Vec::new();
        for _ in 0..RANDOM_VECTORS_PER_ITER {
            let mat = self.optimal_input_mapping_sampler.sample(&mut rng);
            sampled_func_mats.push(mat);
        }

        let num_models = sampled_embeddings.models.len();
        let compressed_func_vec_size = func_feat_info.get_sketched_dimensions();
        let in_vec_size = func_space_info.in_feat_info.get_sketched_dimensions();

        let mut in_model_funcs = Array::zeros((num_models, compressed_func_vec_size));
        let mut out_model_vecs = Array::zeros((num_models, in_vec_size));
        let mut ind = 0;

        for model_key in sampled_embeddings.models.keys() {
            let model_embedding = sampled_embeddings.get_embedding(*model_key);

            let model_compressed_vec = model_embedding.sampled_compressed_vec.view();

            //Featurized model embedding matrix
            let model_feat_vec = func_feat_info.get_features(model_compressed_vec);

            let value_field_max_solver = ValueFieldMaximumSolver {
                func_mat : model_embedding.sampled_mat.clone(),
                value_field : value_field.clone(),
                func_type_id : self.func_type_id
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
                    if (-result.state.cost >= best_value || best_index.is_none()) {
                        best_value = -result.state.cost;
                        best_index = Option::Some(*model_key);
                    }
                    result.state.param
                },
                Err(e) => {
                    error!("Unexpected error on optimization: {}", e);
                    initial_vector.clone()
                }
            };

            in_model_funcs.row_mut(ind).assign(&model_compressed_vec);
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

        if (best_index.is_none()) {
            Option::None
        } else {
            Option::Some((best_index.unwrap(), best_value))
        }
    }
}
