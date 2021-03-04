use ndarray_linalg::*;
use ndarray::*;
use crate::params::*;
use crate::schmeared_hole::*;
use crate::sigma_points::*;
use crate::constraint_collection::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::rc::*;
use crate::vector_application_result::*;
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
use crate::function_space_directory::*;

pub struct ValueFieldState {
    pub value_fields : HashMap<TypeId, ValueField>,
    pub target : SchmearedHole
}

impl ValueFieldState {
    pub fn new(target : SchmearedHole, func_space_directory : &FunctionSpaceDirectory) -> ValueFieldState {
        let mut value_fields = HashMap::new();
        for func_type_id in func_space_directory.directory.keys() {
            let func_space_info = func_space_directory.directory.get(func_type_id).unwrap();
            let func_feat_info = Rc::clone(&func_space_info.func_feat_info);
            let value_field = ValueField::new(func_feat_info);
            value_fields.insert(*func_type_id, value_field);
        }
        ValueFieldState {
            value_fields,
            target
        }
    }
    //Deals in compressed vectors
    pub fn get_value_for_vector(&self, feature_space_info : &FeatureSpaceInfo, typed_vector : &TypedVector) -> f32 {
        let compressed_vec = &typed_vector.vec;
        let feat_vec = feature_space_info.get_features(compressed_vec);
        let type_id = typed_vector.type_id;
        let value_field = self.get_value_field(type_id);
        let additional_value = value_field.get_dot_product(&feat_vec);
        if (type_id == self.target.type_id) {
            let schmear_sq_dist = self.target.compressed_inv_schmear.sq_mahalanobis_dist(&compressed_vec);
            additional_value - schmear_sq_dist
        } else {
            additional_value
        }
    }
    pub fn get_target_for_type(&self, type_id : TypeId) -> Option<SchmearedHole> {
        if (type_id == self.target.type_id) {
            Option::Some(self.target.clone())
        } else {
            Option::None
        }
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
        let func_feat_info = &self.get_value_field(func_vec.type_id).feat_space_info;
        let ret_feat_info = &self.get_value_field(ret_vec.type_id).feat_space_info;

        let func_feat_vec = func_vec.get_features_from_base(func_feat_info);
        let ret_feat_vec = ret_vec.get_features_from_base(ret_feat_info);

        let mut bonus = 0.0f32;
        //If the return type is the target type, add in the bonus
        if (self.target.type_id == ret_vec.type_id) {
            let sq_dist = self.target.full_inv_schmear.sq_mahalanobis_dist(&ret_vec.vec);
            bonus = -sq_dist;
        }

        if (is_vector_type(arg_vec.type_id)) {
            self.apply_vector_arg_feat_constraint(GAMMA, LAMBDA, &func_feat_vec, &ret_feat_vec, bonus);
        } else {
            let arg_feat_info = &self.get_value_field(arg_vec.type_id).feat_space_info;
            let arg_feat_vec = arg_vec.get_features_from_base(arg_feat_info);

            self.apply_function_arg_feat_constraint(GAMMA, LAMBDA, &func_feat_vec, &arg_feat_vec,
                                                    &ret_feat_vec, bonus);
        }
    }

    pub fn apply_vector_arg_feat_constraint(&mut self, gamma : f32, lambda : f32,
                          func_vec : &TypedVector,
                          ret_vec : &TypedVector, bonus : f32) {
        let func_field = self.get_value_field(func_vec.type_id);
        let ret_field = self.get_value_field(ret_vec.type_id);

        //Equations are gamma * func_value = ret_value + bonus
        let func_coef = gamma * &func_vec.vec;
        let ret_coef = -1.0f32 * &ret_vec.vec;

        let func_dot = func_field.get_dot_product(&func_coef);
        let ret_dot = ret_field.get_dot_product(&ret_coef);
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
