use ndarray::*;
use ndarray_linalg::*;
use crate::space_info::*;
use std::rc::*;
use std::collections::HashMap;

type ModelKey = usize;

pub struct SampledEmbeddingSpace {
    pub space_info : Rc<SpaceInfo>,
    models : HashMap<ModelKey, Array2<f32>>
}

impl SampledEmbeddingSpace {
    pub fn has_embedding(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
    pub fn get_embedding(&self, model_key : ModelKey) -> &Array2<f32> {
        self.models.get(&model_key).unwrap()
    }
    pub fn get_compressed_embedding(&self, model_key : ModelKey) -> Array1<f32> {
        let full_embedding = self.get_embedding(model_key);
        let full_dim = full_embedding.shape()[0] * full_embedding.shape()[1];
        let reshaped_embedding = full_embedding.clone().into_shape((full_dim,)).unwrap();
        self.space_info.func_sketcher.sketch(&reshaped_embedding)
    }
}
