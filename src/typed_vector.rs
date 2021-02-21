use ndarray_linalg::*;
use ndarray::*;
use crate::type_id::*;
use crate::feature_space_info::*;

pub struct TypedVector {
    pub type_id : TypeId, 
    pub vec : Array1<f32>
}

impl TypedVector {
    pub fn get_features_from_base(&self, feat_space_info : &FeatureSpaceInfo) -> TypedVector {
        let vec = feat_space_info.get_features_from_base(&self.vec);
        TypedVector {
            type_id : self.type_id,
            vec
        }
    }
}
