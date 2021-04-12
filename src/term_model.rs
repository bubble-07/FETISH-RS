extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::params::*;
use crate::data_update::*;
use crate::space_info::*;
use crate::type_id::*;
use crate::normal_inverse_wishart::*;
use crate::term_application::*;
use crate::term_reference::*;
use crate::func_schmear::*;
use crate::prior_specification::*;
use crate::func_inverse_schmear::*;
use crate::context::*;

use rand::prelude::*;

use std::collections::HashMap;
use crate::model::*;

#[derive(Clone)]
pub struct TermModel<'a> {
    pub model : Model<'a>,
    prior_updates : HashMap::<TermApplication, NormalInverseWishart>,
    pub data_updates : HashMap::<TermReference, DataUpdate>
}

pub struct TermModelPriorSpecification {
}

impl PriorSpecification for TermModelPriorSpecification {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        TERM_MODEL_IN_PRECISION_MULTIPLIER
    }
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32 {
        let pseudo_observations = self.get_out_pseudo_observations(out_dims);
        pseudo_observations * TERM_MODEL_OUT_COVARIANCE_MULTIPLIER
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        //Minimal number of pseudo-observations to have a defined
        //mean and covariance with no observations on the model
        (out_dims as f32) + 4.0f32
    }
}

impl <'a> TermModel<'a> {
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

    pub fn get_features(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        let func_space_info = self.model.ctxt.get_function_space_info(self.model.get_type_id());
        func_space_info.in_feat_info.get_features(in_vec)
    }

    pub fn eval(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        self.model.eval(in_vec)
    }

    pub fn has_data(&self, update_key : &TermReference) -> bool {
        self.data_updates.contains_key(update_key)
    }
    pub fn update_data(&mut self, update_key : TermReference, data_update : DataUpdate) {
        let func_space_info = self.model.ctxt.get_function_space_info(self.model.get_type_id());
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

    pub fn new(type_id : TypeId, ctxt : &'a Context) -> TermModel<'a> {
        let prior_updates : HashMap::<TermApplication, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<TermReference, DataUpdate> = HashMap::new();
        let arg_type_id = ctxt.get_arg_type_id(type_id);
        let ret_type_id = ctxt.get_ret_type_id(type_id);

        let prior_specification = TermModelPriorSpecification {};

        let model = Model::new(&prior_specification, arg_type_id, ret_type_id, ctxt);
        TermModel {
            model : model,
            prior_updates : prior_updates,
            data_updates : data_updates
        }
    }
}
