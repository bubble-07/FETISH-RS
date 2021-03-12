extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::function_space_info::*;
use crate::data_update::*;
use crate::space_info::*;
use crate::data_point::*;
use crate::pseudoinverse::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::rand_utils::*;
use crate::type_id::*;
use crate::enum_feature_collection::*;
use crate::linalg_utils::*;
use crate::normal_inverse_wishart::*;
use crate::term_application::*;
use crate::func_scatter_tensor::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::params::*;
use crate::test_utils::*;

use rand::prelude::*;

use std::collections::HashMap;
use crate::model::*;

#[derive(Clone)]
pub struct TermModel {
    pub model : Model,
    prior_updates : HashMap::<TermApplication, NormalInverseWishart>,
    pub data_updates : HashMap::<TermReference, DataUpdate>
}

impl TermModel {
    pub fn get_total_dims(&self) -> usize {
        self.model.get_total_dims()
    }
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.model.sample(rng)
    }
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        self.model.sample_as_vec(rng)
    }
    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        self.model.get_mean_as_vec()
    }

    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        self.model.get_inverse_schmear()
    }

    pub fn get_schmear(&self) -> FuncSchmear {
        self.model.get_schmear()
    }

    pub fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        let func_space_info = get_function_space_info(self.model.get_type_id());
        func_space_info.in_feat_info.get_features(in_vec)
    }

    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        self.model.eval(in_vec)
    }

    pub fn has_data(&self, update_key : &TermReference) -> bool {
        self.data_updates.contains_key(update_key)
    }
    pub fn update_data(&mut self, update_key : TermReference, data_update : DataUpdate) {
        let func_space_info = get_function_space_info(self.model.get_type_id());
        let feat_update = data_update.featurize(&func_space_info);
        self.model.data += &feat_update;
        self.data_updates.insert(update_key, feat_update);
    }
    pub fn downdate_data(&mut self, update_key : &TermReference) {
        let data_update = self.data_updates.remove(update_key).unwrap();
        self.model.data -= &data_update;
    }    
    pub fn has_prior(&self, update_key : &TermApplication) -> bool {
        self.prior_updates.contains_key(update_key)
    }
    pub fn update_prior(&mut self, update_key : TermApplication, distr : NormalInverseWishart) {
        self.model += &distr;
        self.prior_updates.insert(update_key, distr);
    }
    pub fn downdate_prior(&mut self, key : &TermApplication) {
        let distr = self.prior_updates.remove(key).unwrap();
        self.model -= &distr;
    }

    pub fn new(type_id : TypeId) -> TermModel {
        let prior_updates : HashMap::<TermApplication, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<TermReference, DataUpdate> = HashMap::new();
        let arg_type_id = get_arg_type_id(type_id);
        let ret_type_id = get_ret_type_id(type_id);
        let model = Model::new(arg_type_id, ret_type_id);
        TermModel {
            model : model,
            prior_updates : prior_updates,
            data_updates : data_updates
        }
    }
}
