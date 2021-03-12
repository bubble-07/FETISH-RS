use ndarray_linalg::*;
use ndarray::*;
use crate::sigma_points::*;
use crate::array_utils::*;
use noisy_float::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
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
use crate::normal_inverse_wishart::*;
use crate::embedder_state::*;
use crate::params::*;
use crate::type_id::*;
use crate::space_info::*;

pub struct ValueField {
    pub coefs : Array1<f32>,
    pub type_id : TypeId
}

impl ValueField {
    //Operates on feature vec
    pub fn get_dot_product(&self, vec : &Array1<f32>) -> f32 {
        self.coefs.dot(vec) 
    }

    pub fn update_coefs(&mut self, delta : &Array1<f32>) {
        self.coefs += delta;
    }

    pub fn new(type_id : TypeId) -> ValueField {
        let feat_space_info = get_feature_space_info(type_id);
        let n = feat_space_info.feature_dimensions; 
        let mut coefs = Array::random((n,), StandardNormal);
        coefs *= INITIAL_VALUE_FIELD_VARIANCE;
        ValueField {
            coefs,
            type_id
        }
    }
}
