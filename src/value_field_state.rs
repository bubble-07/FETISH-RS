use ndarray_linalg::*;
use ndarray::*;
use crate::params::*;
use crate::schmeared_hole::*;
use crate::sigma_points::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::rc::*;
use crate::feature_space_info::*;
use crate::sampled_embedder_state::*;
use crate::function_space_info::*;
use crate::data_update::*;
use crate::data_point::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use crate::term_model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::feature_collection::*;
use crate::enum_feature_collection::*;
use crate::vector_space::*;
use crate::normal_inverse_wishart::*;
use crate::embedder_state::*;
use crate::value_field::*;
use crate::typed_vector::*;

pub struct ValueFieldState {
    pub value_fields : HashMap<TypeId, ValueField>,
    pub target : SchmearedHole
}

impl ValueFieldState {
    //Assumes that we're dealing with base type vectors
    pub fn apply_constraint(&mut self, func_vec : &TypedVector, arg_vec : &TypedVector, ret_vec : &TypedVector) {
        let func_feat_info = &self.get_value_field(func_vec.type_id).feat_space_info;
        let arg_feat_info = &self.get_value_field(arg_vec.type_id).feat_space_info;
        let ret_feat_info = &self.get_value_field(ret_vec.type_id).feat_space_info;

        let func_feat_vec = func_vec.get_features_from_base(func_feat_info);
        let mut arg_feat_vec = arg_vec.get_features_from_base(arg_feat_info);
        let ret_feat_vec = ret_vec.get_features_from_base(ret_feat_info);

        //If the argument is a vector type, then zero out the feature vector
        //since we'll want to drop it from the equation
        if (is_vector_type(arg_vec.type_id)) {
            arg_feat_vec.vec = Array::zeros((arg_feat_vec.vec.shape()[0],));
        }

        let mut bonus = 0.0f32;
        //If the return type is the target type, add in the bonus
        if (self.target.type_id == ret_vec.type_id) {
            let sq_dist = self.target.full_inv_schmear.sq_mahalanobis_dist(&ret_vec.vec);
            bonus = -sq_dist;
        }
        self.apply_feat_constraint(GAMMA, LAMBDA, 
                                   &func_feat_vec, &arg_feat_vec, 
                                   &ret_feat_vec, bonus);
    }
    

    //Assumes that we're dealing with _featurized_ type vectors
    //lambda is the relaxation factor for the Kaczmarz method
    pub fn apply_feat_constraint(&mut self, gamma : f32, lambda : f32,
                          func_vec : &TypedVector, arg_vec : &TypedVector, 
                          ret_vec : &TypedVector, bonus : f32) {
        let func_field = self.get_value_field(func_vec.type_id);
        let arg_field = self.get_value_field(arg_vec.type_id);
        let ret_field = self.get_value_field(ret_vec.type_id);
        
        //Equations are gamma * func_value + gamma * arg_value = ret_value + bonus
        
        let func_coef = gamma * &func_vec.vec;
        let arg_coef = gamma * &arg_vec.vec;
        let ret_coef = -1.0f32 * &ret_vec.vec;

        let func_dot = func_field.get_dot_product(&func_coef);
        let arg_dot = arg_field.get_dot_product(&arg_coef);
        let ret_dot = ret_field.get_dot_product(&ret_coef);
        let total_dot = func_dot + arg_dot + ret_dot;

        let total_sq_magnitude = func_coef.dot(&func_coef) + arg_coef.dot(&arg_coef) + ret_coef.dot(&ret_coef);
        
        let coef_update_magnitude = lambda * (bonus - total_dot) / total_sq_magnitude;

        let func_update = coef_update_magnitude * &func_coef;
        let arg_update = coef_update_magnitude * &arg_coef;
        let ret_update = coef_update_magnitude * &ret_coef;

        self.get_value_field_mut(func_vec.type_id).update_coefs(&func_update);
        self.get_value_field_mut(arg_vec.type_id).update_coefs(&arg_update);
        self.get_value_field_mut(ret_vec.type_id).update_coefs(&ret_update);
    }

    pub fn get_value_field(&self, type_id : TypeId) -> &ValueField {
        self.value_fields.get(&type_id).unwrap()
    }
    pub fn get_value_field_mut(&mut self, type_id : TypeId) -> &mut ValueField {
        self.value_fields.get_mut(&type_id).unwrap()
    }

}
