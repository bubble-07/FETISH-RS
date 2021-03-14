use ndarray::*;
use crate::array_utils::*;
use crate::type_id::*;
use crate::space_info::*;

#[derive(Clone)]
pub enum SampledTermEmbedding {
    VectorEmbedding(Array1<f32>),
    FunctionEmbedding(TypeId, Array2<f32>)
}

impl SampledTermEmbedding {
    pub fn get_flattened(&self) -> Array1<f32> {
        match (&self) {
            SampledTermEmbedding::VectorEmbedding(vec) => vec.clone(),
            SampledTermEmbedding::FunctionEmbedding(_, full_embedding) => flatten_matrix(full_embedding)
        }
    }
    pub fn get_compressed(&self) -> Array1<f32> {
        let flattened_embedding = self.get_flattened();
        match (&self) {
            SampledTermEmbedding::VectorEmbedding(_) => flattened_embedding,
            SampledTermEmbedding::FunctionEmbedding(type_id, _) => {
                let func_feat_info = get_feature_space_info(*type_id);
                func_feat_info.sketch(&flattened_embedding)
            }
        }
    }
}
