use ndarray::*;

use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use crate::sampled_embedder_state::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
///An [`InverseSchmear`] which has been compressed relative to some expansion matrix.
///When an inverse schmear is compressed, it may be in a subspace such that
///there's always a positive offset from zero in mahalanobis distance, hence the constant
///`extra_sq_distance` included in this struct's [`sq_mahalanobis_dist`] implementation.
///See [`InverseSchmear#compress`].
pub struct CompressedInverseSchmear {
    pub inv_schmear : InverseSchmear,
    pub extra_sq_distance : f32
}

impl CompressedInverseSchmear {
    ///Returns the squared mahalanobis distance for this [`CompressedInverseSchmear`]
    ///to the given compressed vector. See [`InverseSchmear#compress`].
    pub fn sq_mahalanobis_dist(&self, vec : ArrayView1<f32>) -> f32 {
        let schmear_distance = self.inv_schmear.sq_mahalanobis_dist(vec);
        schmear_distance + self.extra_sq_distance
    }
}
