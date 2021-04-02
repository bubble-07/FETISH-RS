use crate::sampled_value_field::*;
use crate::params::*;
use crate::schmeared_hole::*;
use crate::constraint_collection::*;
use std::collections::HashMap;
use crate::vector_application_result::*;
use crate::type_id::*;
use crate::space_info::*;
use crate::value_field::*;
use crate::typed_vector::*;

pub struct SampledValueFieldState {
    pub sampled_value_fields : HashMap<TypeId, SampledValueField>
}

impl SampledValueFieldState {
    pub fn get_value_field(&self, type_id : TypeId) -> &SampledValueField {
        self.sampled_value_fields.get(&type_id).unwrap()
    }

    pub fn get_value_field_mut(&mut self, type_id : TypeId) -> &mut SampledValueField {
        self.sampled_value_fields.get_mut(&type_id).unwrap()
    }

    pub fn get_value_for_full_vector(&self, typed_vector : &TypedVector) -> f32 {
        let type_id = typed_vector.type_id;
        let value_field = self.get_value_field(type_id);
        let result = value_field.get_value_for_full_vector(&typed_vector.vec);
        result
    }

    pub fn get_value_for_compressed_vector(&self, typed_vector : &TypedVector) -> f32 {
        let type_id = typed_vector.type_id;
        let value_field = self.get_value_field(type_id);
        let result = value_field.get_value_for_compressed_vector(&typed_vector.vec);
        result
    }

    pub fn apply_constraints(&mut self, constraint_collection : &ConstraintCollection) {
        for vec_app_result in &constraint_collection.constraints {
            self.apply_constraint_from_application_result(vec_app_result);
        }
    }

    pub fn apply_constraint_from_application_result(&mut self, vec_app_result : &VectorApplicationResult) {
        let func_vec = vec_app_result.get_func_vec();
        let arg_vec = vec_app_result.get_arg_vec();
        let ret_vec = vec_app_result.get_ret_vec();
        self.apply_constraint(&func_vec, &arg_vec, &ret_vec);
    }

    //Assumes that we're dealing with base type vectors
    pub fn apply_constraint(&mut self, func_vec : &TypedVector, arg_vec : &TypedVector, ret_vec : &TypedVector) {
        let func_feat_info = get_feature_space_info(func_vec.type_id);
        let ret_feat_info = get_feature_space_info(ret_vec.type_id);

        let func_feat_vec = func_vec.get_features_from_base(func_feat_info);
        let ret_feat_vec = ret_vec.get_features_from_base(ret_feat_info);

        let mut bonus = 0.0f32;
        let func_value_field = self.get_value_field(func_vec.type_id);
        let ret_value_field = self.get_value_field(ret_vec.type_id);
        
        bonus += func_value_field.get_schmear_sq_dist_from_full_vec(&func_vec.vec);
        bonus -= ret_value_field.get_schmear_sq_dist_from_full_vec(&ret_vec.vec);

        if (is_vector_type(arg_vec.type_id)) {
            self.apply_vector_arg_feat_constraint(GAMMA, LAMBDA, &func_feat_vec, &ret_feat_vec, bonus);
        } else {
            let arg_feat_info = get_feature_space_info(arg_vec.type_id);
            let arg_feat_vec = arg_vec.get_features_from_base(arg_feat_info);

            let arg_value_field = self.get_value_field(arg_vec.type_id);
            bonus += arg_value_field.get_schmear_sq_dist_from_full_vec(&arg_vec.vec);

            self.apply_function_arg_feat_constraint(GAMMA, LAMBDA, &func_feat_vec, &arg_feat_vec,
                                                    &ret_feat_vec, bonus);
        }
    }

    //Deals in featurized vectors
    pub fn apply_vector_arg_feat_constraint(&mut self, gamma : f32, lambda : f32,
                          func_vec : &TypedVector,
                          ret_vec : &TypedVector, bonus : f32) {
        let func_field = self.get_value_field(func_vec.type_id);
        let ret_field = self.get_value_field(ret_vec.type_id);

        //Equations are gamma * func_value = ret_value + bonus
        let func_coef = gamma * &func_vec.vec;
        let ret_coef = -1.0f32 * &ret_vec.vec;

        let func_dot = func_field.get_dot_product_from_feat_vec(&func_coef);
        let ret_dot = ret_field.get_dot_product_from_feat_vec(&ret_coef);
        let total_dot = func_dot + ret_dot;

        let total_sq_magnitude = func_coef.dot(&func_coef) + ret_coef.dot(&ret_coef);

        let coef_update_magnitude = lambda * (bonus - total_dot) / total_sq_magnitude;

        let func_update = coef_update_magnitude * &func_coef;
        let ret_update = coef_update_magnitude * &ret_coef;

        self.get_value_field_mut(func_vec.type_id).update_coefs(&func_update);
        self.get_value_field_mut(ret_vec.type_id).update_coefs(&ret_update);
    }
    

    //Assumes that we're dealing with _featurized_ type vectors
    //lambda is the relaxation factor for the Kaczmarz method
    pub fn apply_function_arg_feat_constraint(&mut self, gamma : f32, lambda : f32,
                          func_vec : &TypedVector, arg_vec : &TypedVector, 
                          ret_vec : &TypedVector, bonus : f32) {
        let func_field = self.get_value_field(func_vec.type_id);
        let arg_field = self.get_value_field(arg_vec.type_id);
        let ret_field = self.get_value_field(ret_vec.type_id);
        
        //Equations are gamma * func_value + gamma * arg_value = ret_value + bonus
        
        let func_coef = (0.5f32 * gamma) * &func_vec.vec;
        let arg_coef = (0.5f32 * gamma) * &arg_vec.vec;
        let ret_coef = -1.0f32 * &ret_vec.vec;

        let func_dot = func_field.get_dot_product_from_feat_vec(&func_coef);
        let arg_dot = arg_field.get_dot_product_from_feat_vec(&arg_coef);
        let ret_dot = ret_field.get_dot_product_from_feat_vec(&ret_coef);
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
}
