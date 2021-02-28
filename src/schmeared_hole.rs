extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::*;
use ndarray_linalg::*;

use crate::array_utils::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use crate::params::*;
use crate::sampled_embedder_state::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
pub struct SchmearedHole {
    pub type_id : TypeId,
    pub full_inv_schmear : InverseSchmear,
    pub compressed_inv_schmear : InverseSchmear
}

impl SchmearedHole {
    pub fn get_closest_term(&self, embedder_state : &SampledEmbedderState) -> (TermReference, f32) {
        if (is_vector_type(self.type_id)) {
            (TermReference::from(&self.full_inv_schmear.mean), 0.0f32)
        } else {
            let embedding_space = embedder_state.embedding_spaces.get(&self.type_id).unwrap();
            let mut best_dist = f32::INFINITY;
            let mut best_term_id = 0; 
            for term_id in embedding_space.models.keys() {
                let embedding = embedding_space.get_embedding(*term_id);
                let embedding_vec = &embedding.sampled_vec;
                let dist = self.full_inv_schmear.sq_mahalanobis_dist(&embedding_vec);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_term_id = *term_id;
                }
            }
            let term_ptr = TermPointer {
                type_id : self.type_id,
                index : best_term_id
            };
            let term_ref = TermReference::FuncRef(term_ptr);
            (term_ref, best_dist)
        }
    }
}
