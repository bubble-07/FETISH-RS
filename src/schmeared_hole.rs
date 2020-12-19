extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::*;
use ndarray_linalg::*;

use crate::bounded_hole::*;
use crate::array_utils::*;
use crate::holed_application::*;
use crate::bounded_holed_application::*;
use crate::holed_linear_expression::*;
use crate::bounded_holed_linear_expression::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::ellipsoid::*;
use crate::type_id::*;
use crate::featurized_points_directory::*;
use crate::params::*;
use crate::featurization_inverse_directory::*;
use crate::sampled_embedder_state::*;
use crate::inverse_schmear::*;

pub struct SchmearedHole {
    pub type_id : TypeId,
    pub full_inv_schmear : InverseSchmear,
    pub compressed_inv_schmear : InverseSchmear
}

impl SchmearedHole {
    pub fn get_closer_than_closest_term_bound(&self, embedder_state : &SampledEmbedderState) -> BoundedHole {
        let (_, sq_mahalanobis_dist) = self.get_closest_term(embedder_state);

        let full_mean = self.full_inv_schmear.mean.clone();
        let compressed_mean = self.compressed_inv_schmear.mean.clone();

        let scale_fac = 1.0f32 / sq_mahalanobis_dist;
        let full_precision = scale_fac * &self.full_inv_schmear.precision;
        let compressed_precision = scale_fac * &self.compressed_inv_schmear.precision;

        let full_bound = Ellipsoid::new(full_mean, full_precision);
        let compressed_bound = Ellipsoid::new(compressed_mean, compressed_precision);


        BoundedHole {
            type_id : self.type_id,
            compressed_bound : compressed_bound
        }
    }
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
