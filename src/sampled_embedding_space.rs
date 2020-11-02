use ndarray::*;
use ndarray_linalg::*;
use crate::space_info::*;
use std::rc::*;
use std::collections::HashMap;
use crate::sampled_term_embedding::*;

type ModelKey = usize;

pub struct SampledEmbeddingSpace {
    pub space_info : Rc<SpaceInfo>,
    pub models : HashMap<ModelKey, Array2<f32>>
}

impl SampledEmbeddingSpace {
    pub fn has_embedding(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
    pub fn get_embedding(&self, model_key : ModelKey) -> SampledTermEmbedding {
        let full_embedding = self.get_raw_embedding(model_key).clone();
        SampledTermEmbedding::FunctionEmbedding(Rc::clone(&self.space_info), full_embedding)
    }

    pub fn get_raw_embedding(&self, model_key : ModelKey) -> &Array2<f32> {
        self.models.get(&model_key).unwrap()
    }

    pub fn new(space_info : Rc<SpaceInfo>) -> SampledEmbeddingSpace {
        SampledEmbeddingSpace {
            space_info : space_info,
            models : HashMap::new()
        }
    }
}
