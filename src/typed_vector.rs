use ndarray::*;
use crate::type_id::*;
use crate::feature_space_info::*;
use crate::context::*;

///A vector with an associated [`TypeId`].
#[derive(Clone)]
pub struct TypedVector {
    pub type_id : TypeId, 
    pub vec : Array1<f32>
}

impl TypedVector {
    ///Given a [`Context`], and assuming that this vector represents
    ///a vector in the base space for the type, returns a [`TypedVector`]
    ///of the same type, but whose vector has been featurized.
    pub fn get_features_from_base(&self, ctxt : &Context) -> TypedVector {
        let feat_space_info = ctxt.get_feature_space_info(self.type_id);
        let vec = feat_space_info.get_features_from_base(self.vec.view());
        TypedVector {
            type_id : self.type_id,
            vec
        }
    }
}
