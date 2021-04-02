use ndarray::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use crate::value_field::*;
use crate::compressed_inv_schmear::*;
use crate::type_id::*;
use crate::normal_inverse_wishart::*;
use crate::params::*;
use crate::model::*;
use crate::term_model::*;
use crate::schmeared_hole::*;
use crate::space_info::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
pub struct SampledValueField {
    pub value_field : ValueField,
    pub compressed_prior_inv_schmear : CompressedInverseSchmear
}

impl SampledValueField {
    pub fn get_feat_vec_coefs(&self) -> &Array1<f32> {
        &self.value_field.feat_vec_coefs
    }

    pub fn get_type_id(&self) -> TypeId {
        self.value_field.get_type_id()
    }

    pub fn update_coefs(&mut self, delta : &Array1<f32>) {
        self.value_field.feat_vec_coefs += delta;
    }

    pub fn get_dot_product_from_feat_vec(&self, feat_vec : &Array1<f32>) -> f32 {
        self.value_field.get_dot_product_from_feat_vec(feat_vec)
    }

    pub fn get_schmear_sq_dist_from_compressed_vec(&self, compressed_vec : &Array1<f32>) -> f32 {
        self.compressed_prior_inv_schmear.sq_mahalanobis_dist(compressed_vec)
    }

    pub fn get_schmear_sq_dist_from_full_vec(&self, full_vec : &Array1<f32>) -> f32 {
        self.value_field.get_schmear_sq_dist_from_full_vec(full_vec)
    }
    
    pub fn get_value_for_full_vector(&self, full_vec : &Array1<f32>) -> f32 {
        self.value_field.get_value_for_full_vector(full_vec)
    }

    pub fn get_value_for_compressed_vector(&self, compressed_vec : &Array1<f32>) -> f32 {
        let type_id = self.get_type_id();
        let feature_space_info = get_feature_space_info(type_id);
        let feat_vec = feature_space_info.get_features(compressed_vec);

        let additional_value = self.get_dot_product_from_feat_vec(&feat_vec);

        let schmear_sq_dist = self.get_schmear_sq_dist_from_compressed_vec(compressed_vec);

        additional_value - schmear_sq_dist
    }
}
