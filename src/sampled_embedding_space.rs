use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;
use std::collections::HashMap;
use crate::sampled_term_embedding::*;
use crate::sampled_model_embedding::*;
use crate::function_space_info::*;
use crate::term_reference::*;
use crate::value_field::*;
use crate::value_field_state::*;
use crate::typed_vector::*;
use crate::type_id::*;

type ModelKey = usize;

pub struct SampledEmbeddingSpace {
    pub func_space_info : FunctionSpaceInfo,
    pub models : HashMap<ModelKey, SampledModelEmbedding>
}

impl SampledEmbeddingSpace {
    pub fn has_embedding(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
    pub fn get_embedding(&self, model_key : ModelKey) -> &SampledModelEmbedding {
        self.models.get(&model_key).unwrap()
    }
    pub fn get_term_embedding(&self, model_key : ModelKey) -> SampledTermEmbedding {
        let space_info = self.func_space_info.clone();
        let model_embedding = self.get_embedding(model_key);
        let sampled_mat = model_embedding.sampled_mat.clone();

        SampledTermEmbedding::FunctionEmbedding(space_info, sampled_mat)
    }
    pub fn get_best_term_index_to_apply_with_value(&self, featurized_arg_vector : &Array1<f32>, 
                                         ret_type : TypeId, value_field_state : &ValueFieldState) 
                                         -> (usize, f32) {
        let mut best_model_index = 0;
        let mut best_model_value = f32::NEG_INFINITY;

        for (model_index, model) in self.models.iter() {
            let mat = &model.sampled_mat;
            let compressed_ret_vec = mat.dot(featurized_arg_vector);
            let typed_ret_vec = TypedVector {
                vec : compressed_ret_vec,
                type_id : ret_type
            };
            let value = value_field_state.get_value_for_vector(&typed_ret_vec);
            if (value > best_model_value) {
                best_model_value = value;
                best_model_index = *model_index;
            }
        }
        (best_model_index, best_model_value)
    }

    pub fn new(func_space_info : FunctionSpaceInfo) -> SampledEmbeddingSpace {
        SampledEmbeddingSpace {
            func_space_info,
            models : HashMap::new()
        }
    }
}
