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
use crate::sampled_value_field::*;
use crate::sampled_embedding_space::*;
use crate::displayable_with_context::*;
use crate::context::*;

#[derive(Clone)]
pub struct ValueField<'a> {
    pub feat_vec_coefs : Array1<f32>,
    pub full_prior_schmear : SchmearedHole,
    pub ctxt : &'a Context
}

impl <'a> ValueField<'a> {
    pub fn get_context(&self) -> &Context {
        self.ctxt
    }
    pub fn sample(&self, sampled_embedding_space : &SampledEmbeddingSpace<'a>) -> SampledValueField<'a> {
        let elaborator = sampled_embedding_space.elaborator.view();
        let full_prior_inv_schmear = &self.full_prior_schmear.inv_schmear;
        let compressed_prior_inv_schmear = full_prior_inv_schmear.compress(elaborator);

        SampledValueField {
            value_field : self.clone(),
            compressed_prior_inv_schmear
        }
    }

    pub fn update_from_sampled(&mut self, sampled_value_field : SampledValueField) {
        self.feat_vec_coefs = sampled_value_field.value_field.feat_vec_coefs;
    }

    pub fn get_type_id(&self) -> TypeId {
        self.full_prior_schmear.type_id
    }

    //Operates on feature vec
    pub fn get_dot_product_from_feat_vec(&self, feat_vec : ArrayView1<f32>) -> f32 {
        self.feat_vec_coefs.dot(&feat_vec) 
    }
    pub fn get_schmear_sq_dist_from_full_vec(&self, full_vec : ArrayView1<f32>) -> f32 {
        self.full_prior_schmear.inv_schmear.sq_mahalanobis_dist(full_vec)
    }

    //Deals in full vectors
    pub fn get_value_for_full_vector(&self, full_vec : ArrayView1<f32>) -> f32 {
        let type_id = self.get_type_id();
        let feature_space_info = self.ctxt.get_feature_space_info(type_id);
        let feat_vec = feature_space_info.get_features_from_base(full_vec);

        let additional_value = self.get_dot_product_from_feat_vec(feat_vec.view());
        
        let schmear_sq_dist = self.get_schmear_sq_dist_from_full_vec(full_vec);

        let result = additional_value - schmear_sq_dist;

        if (!result.is_finite()) {
            error!("Non-finite value for type {} and vec {}", 
                self.ctxt.get_type(type_id).display(self.ctxt), full_vec);
        }
        result
    }

    pub fn from_type_id(type_id : TypeId, ctxt : &'a Context) -> ValueField<'a> {
        let default_model = TermModel::new(type_id, ctxt);

        let full_prior_schmear = default_model.get_schmeared_hole().rescale_spread(TARGET_INV_SCHMEAR_SCALE_FAC);
        ValueField::new(full_prior_schmear, ctxt)
    }

    pub fn new(full_prior_schmear : SchmearedHole, ctxt : &'a Context) -> ValueField<'a> {
        let feat_space_info = ctxt.get_feature_space_info(full_prior_schmear.type_id);
        let n = feat_space_info.feature_dimensions; 
        let mut feat_vec_coefs = Array::random((n,), StandardNormal);
        feat_vec_coefs *= INITIAL_VALUE_FIELD_VARIANCE;
        ValueField {
            feat_vec_coefs,
            full_prior_schmear,
            ctxt
        }
    }
}
