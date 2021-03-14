use ndarray::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use crate::type_id::*;
use crate::params::*;
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
