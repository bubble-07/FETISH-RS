use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use crate::sampled_embedder_state::*;
use crate::inverse_schmear::*;
use crate::context::*;
use crate::array_utils::*;

#[derive(Clone)]
pub struct SchmearedHole {
    pub type_id : TypeId,
    pub inv_schmear : InverseSchmear
}

impl SchmearedHole {
    pub fn rescale_spread(&self, scale_fac : f32) -> SchmearedHole {
        let inv_schmear = self.inv_schmear.rescale_spread(scale_fac);
        SchmearedHole {
            type_id : self.type_id,
            inv_schmear
        }
    }
    pub fn get_closest_term(&self, embedder_state : &SampledEmbedderState) -> (TermReference, f32) {
        if (embedder_state.ctxt.is_vector_type(self.type_id)) {
            (TermReference::VecRef(self.type_id, to_noisy(&self.inv_schmear.mean)), 0.0f32)
        } else {
            let embedding_space = embedder_state.embedding_spaces.get(&self.type_id).unwrap();
            let mut best_dist = f32::INFINITY;
            let mut best_term_id = Option::None;
            for term_id in embedding_space.models.keys() {
                let embedding = embedding_space.get_embedding(*term_id);
                let embedding_vec = &embedding.sampled_vec;
                let dist = self.inv_schmear.sq_mahalanobis_dist(&embedding_vec);
                if (dist <= best_dist || best_term_id.is_none()) {
                    best_dist = dist;
                    best_term_id = Option::Some(*term_id);
                }
            }
            let term_ptr = TermPointer {
                type_id : self.type_id,
                index : best_term_id.unwrap()
            };
            let term_ref = TermReference::FuncRef(term_ptr);
            (term_ref, best_dist)
        }
    }
}
