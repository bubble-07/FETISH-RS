use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;

pub struct SampledEmbedderState {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace>
}
