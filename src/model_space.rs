extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use std::ops;
use std::rc::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::model::*;
use crate::bayes_utils::*;
use crate::schmear::*;
use arraymap::ArrayMap;

use std::collections::HashMap;

type ModelKey = usize;

pub struct ModelSpace {
    in_dimensions : usize,
    feature_dimensions : usize,
    out_dimensions : usize,
    feature_collections : Rc<[EnumFeatureCollection; 3]>,
    models : HashMap<ModelKey, Model>
}

impl ModelSpace {

    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    pub fn apply_schmears(&self, f : &Schmear, x : &Schmear) -> Schmear {
        let (f_mean, f_covar) = schmear_to_tensors(self.out_dimensions, self.feature_dimensions, f);
        self.compute_out_schmear(&f_mean, &f_covar, x)
    }

    fn compute_out_schmear(&self, f_mean : &Array2<f32>, f_covar : &Array4<f32>,
                           x : &Schmear) -> Schmear {
        let x_mean = &x.mean;
        let x_covar = &x.covariance;

        let feat_vec = self.get_features(&x_mean);
        let jacobian = self.get_jacobian(&x_mean);

        //There are two terms here for covariance -- J_f(x)^T sigma_x J_f(x)^T
        let data_contrib = einsum("ts,sr,qr->tq", &[&jacobian, x_covar, &jacobian])
                            .unwrap().into_dimensionality::<Ix2>().unwrap();

        
        //and the double-contraction of sigma_f by featurized x's
        let model_contrib = einsum("tsrq,s,q->tr", &[f_covar, &feat_vec, &feat_vec])
                            .unwrap().into_dimensionality::<Ix2>().unwrap();
        
        let out_covar = data_contrib + model_contrib;
        let out_mean = einsum("ab,b->a", &[f_mean, x_mean])
                            .unwrap().into_dimensionality::<Ix1>().unwrap();
        Schmear {
            mean : out_mean,
            covariance : out_covar
        }
    }

    pub fn add_model(&mut self, model_key : ModelKey) {
        let model = Model::new(Rc::clone(&self.feature_collections), self.in_dimensions, self.out_dimensions);
        self.models.insert(model_key, model);
    }
    
    pub fn get_model_mut(&mut self, model_key : ModelKey) -> &mut Model {
        self.models.get_mut(&model_key).unwrap()
    }
    pub fn get_model(&self, model_key : ModelKey) -> &Model {
        self.models.get(&model_key).unwrap()
    }
}
