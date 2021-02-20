use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;
use std::collections::HashMap;
use std::rc::*;
use crate::function_space_info::*;
use crate::term_pointer::*;
use crate::type_id::*;
use crate::sampled_term_embedding::*;
use crate::term_reference::*;
use crate::array_utils::*;
use crate::sampled_model_embedding::*;

pub struct SampledEmbedderState {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace>
}

impl SampledEmbedderState {
    pub fn has_embedding(&self, term_ptr : &TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    pub fn get_model_embedding(&self, term_ptr : &TermPointer) -> &SampledModelEmbedding {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_embedding(term_ptr.index)
    }
    pub fn get_space_info(&self, type_id : &TypeId) -> &FunctionSpaceInfo {
        let result = &self.embedding_spaces.get(type_id).unwrap().func_space_info;
        result
    }

    fn inflate_embedding(&self, type_id : TypeId, compressed_embedding : Array1<f32>) -> SampledTermEmbedding {
        if (is_vector_type(type_id)) {
            SampledTermEmbedding::VectorEmbedding(compressed_embedding)
        } else {
            let space_info = self.get_space_info(&type_id);
            let inflated = space_info.inflate_compressed_vector(&compressed_embedding);
            SampledTermEmbedding::FunctionEmbedding(space_info.clone(), inflated)
        }
    }

    pub fn get_term_embedding(&self, term_ref : &TermReference) -> SampledTermEmbedding {
        match (term_ref) {
            TermReference::VecRef(vec) => SampledTermEmbedding::VectorEmbedding(from_noisy(vec)),
            TermReference::FuncRef(term_ptr) => {
                let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
                space.get_term_embedding(term_ptr.index)
            }
        }
    }
}
