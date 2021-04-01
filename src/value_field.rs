use ndarray::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use crate::type_id::*;
use crate::normal_inverse_wishart::*;
use crate::params::*;
use crate::model::*;
use crate::term_model::*;
use crate::schmeared_hole::*;
use crate::space_info::*;

#[derive(Clone)]
pub struct ValueField {
    pub coefs : Array1<f32>,
    pub prior_schmear : SchmearedHole
}

impl ValueField {
    pub fn get_type_id(&self) -> TypeId {
        self.prior_schmear.type_id
    }

    //Operates on feature vec
    pub fn get_dot_product(&self, feat_vec : &Array1<f32>) -> f32 {
        self.coefs.dot(feat_vec) 
    }

    pub fn update_coefs(&mut self, delta : &Array1<f32>) {
        self.coefs += delta;
    }

    pub fn get_schmear_sq_dist(&self, compressed_vec : &Array1<f32>) -> f32 {
        self.prior_schmear.compressed_inv_schmear.sq_mahalanobis_dist(compressed_vec)
    }

    //Deals in compressed vectors
    pub fn get_value_for_vector(&self, compressed_vec : &Array1<f32>) -> f32 {
        let type_id = self.get_type_id();
        let feature_space_info = get_feature_space_info(type_id);
        let feat_vec = feature_space_info.get_features(compressed_vec);

        let additional_value = self.get_dot_product(&feat_vec);
        
        let schmear_sq_dist = self.get_schmear_sq_dist(compressed_vec);

        additional_value - schmear_sq_dist
    }

    pub fn from_type_id(type_id : TypeId) -> ValueField {
        let prior_specification = TermModelPriorSpecification { };

        let arg_type = get_arg_type_id(type_id);
        let ret_type = get_ret_type_id(type_id);
        
        let default_model = Model::new(&prior_specification, arg_type, ret_type);

        let prior_schmear = default_model.get_schmeared_hole().rescale_spread(TARGET_INV_SCHMEAR_SCALE_FAC);
        ValueField::new(prior_schmear)
    }

    pub fn new(prior_schmear : SchmearedHole) -> ValueField {
        let feat_space_info = get_feature_space_info(prior_schmear.type_id);
        let n = feat_space_info.feature_dimensions; 
        let mut coefs = Array::random((n,), StandardNormal);
        coefs *= INITIAL_VALUE_FIELD_VARIANCE;
        ValueField {
            coefs,
            prior_schmear
        }
    }
}
