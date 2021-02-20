use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;
use std::collections::HashMap;
use crate::sampled_term_embedding::*;
use crate::sampled_model_embedding::*;
use crate::function_space_info::*;

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
    pub fn new(func_space_info : FunctionSpaceInfo) -> SampledEmbeddingSpace {
        SampledEmbeddingSpace {
            func_space_info,
            models : HashMap::new()
        }
    }
}
