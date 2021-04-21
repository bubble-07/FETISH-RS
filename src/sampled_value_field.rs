use fetish_lib::everything::*;
use ndarray::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use crate::params::*;
use crate::term_model::*;
use crate::value_field::*;

#[derive(Clone)]
pub struct SampledValueField<'a> {
    pub value_field : ValueField<'a>,
    pub compressed_prior_inv_schmear : CompressedInverseSchmear,
}

impl<'a> SampledValueField<'a> {
    pub fn get_context(&self) -> &Context {
        self.value_field.get_context()
    }

    pub fn get_feat_vec_coefs(&self) -> &Array1<f32> {
        &self.value_field.feat_vec_coefs
    }

    pub fn get_type_id(&self) -> TypeId {
        self.value_field.get_type_id()
    }

    pub fn update_coefs(&mut self, delta : ArrayView1<f32>) {
        self.value_field.feat_vec_coefs += &delta;
    }

    pub fn get_dot_product_from_feat_vec(&self, feat_vec : ArrayView1<f32>) -> f32 {
        self.value_field.get_dot_product_from_feat_vec(feat_vec)
    }

    pub fn get_schmear_sq_dist_from_compressed_vec(&self, compressed_vec : ArrayView1<f32>) -> f32 {
        self.compressed_prior_inv_schmear.sq_mahalanobis_dist(compressed_vec)
    }

    pub fn get_schmear_sq_dist_from_full_vec(&self, full_vec : ArrayView1<f32>) -> f32 {
        self.value_field.get_schmear_sq_dist_from_full_vec(full_vec)
    }
    
    pub fn get_value_for_full_vector(&self, full_vec : ArrayView1<f32>) -> f32 {
        self.value_field.get_value_for_full_vector(full_vec)
    }

    pub fn get_value_for_compressed_vector(&self, compressed_vec : ArrayView1<f32>) -> f32 {
        let type_id = self.get_type_id();
        let feature_space_info = self.get_context().get_feature_space_info(type_id);
        let feat_vec = feature_space_info.get_features(compressed_vec);

        let additional_value = self.get_dot_product_from_feat_vec(feat_vec.view());

        let schmear_sq_dist = self.get_schmear_sq_dist_from_compressed_vec(compressed_vec);

        let result = additional_value - schmear_sq_dist;

        if (!result.is_finite()) {
            error!("Non-finite value for type {} and vec {}", 
                self.get_context().get_type(type_id).display(self.get_context()), 
                compressed_vec);
        }
        result
    }
}
