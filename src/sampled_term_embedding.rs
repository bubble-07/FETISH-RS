use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;
use crate::space_info::*;
use crate::array_utils::*;

#[derive(Clone)]
pub enum SampledTermEmbedding {
    VectorEmbedding(Array1<f32>),
    FunctionEmbedding(Rc<SpaceInfo>, Array2<f32>)
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
            SampledTermEmbedding::VectorEmbedding(vec) => flattened_embedding,
            SampledTermEmbedding::FunctionEmbedding(space_info, _) => {
                space_info.sketch(&flattened_embedding)
            }
        }
    }
}
