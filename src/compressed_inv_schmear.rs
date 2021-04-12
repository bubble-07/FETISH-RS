use ndarray::*;

use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use crate::sampled_embedder_state::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
//When an inverse schmear is compressed, it may be in a subspace such that
//there's always a positive offset from zero in mahalanobis distance, hence the constant
pub struct CompressedInverseSchmear {
    pub inv_schmear : InverseSchmear,
    pub extra_sq_distance : f32
}

impl CompressedInverseSchmear {
    pub fn sq_mahalanobis_dist(&self, vec : ArrayView1<f32>) -> f32 {
        let schmear_distance = self.inv_schmear.sq_mahalanobis_dist(vec);
        schmear_distance + self.extra_sq_distance
    }
}
