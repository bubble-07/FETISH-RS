use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;
use std::collections::HashMap;
use crate::term_pointer::*;
use crate::type_id::*;

pub struct SampledEmbedderState {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace>
}

impl SampledEmbedderState {
    pub fn has_embedding(&self, term_ptr : &TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    pub fn get_embedding(&self, term_ptr : &TermPointer) -> &Array2<f32> {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_embedding(term_ptr.index)
    }
    pub fn get_compressed_embedding(&self, term_ptr : &TermPointer) -> Array1<f32> {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_compressed_embedding(term_ptr.index)
    }
}
