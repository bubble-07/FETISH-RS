extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::rc::*;
use crate::least_squares::*;
use crate::model::*;
use crate::params::*;
use crate::test_utils::*;
use crate::inverse_schmear::*;
use crate::enum_feature_collection::*;

extern crate pretty_env_logger;


#[derive(Clone)]
pub struct SampledFunction {
    pub in_dimensions : usize,
    pub mat : Array2<f32>,
    pub feature_collections : Vec<EnumFeatureCollection>
}

impl SampledFunction {
    pub fn apply(&self, input : &Array1<f32>) -> Array1<f32> {
        let features : Array1<f32> = to_features(&self.feature_collections, input);
        let result : Array1<f32> = self.mat.dot(&features);
        result
    }
    pub fn jacobian(&self, input : &Array1<f32>) -> Array2<f32> {
        let feat_jacobian = to_jacobian(&self.feature_collections, input);
        let result : Array2<f32> = self.mat.dot(&feat_jacobian);
        result
    }

    //Yields a new input, which according to the secant method, should be better
    //than the passed input for achieving the
    pub fn secant_method_iter(&self, x : &Array1<f32>, target : &InverseSchmear) -> Array1<f32> {
        let y = self.apply(x);
        let J = self.jacobian(x);
        let delta_y = &target.mean - &y;
        let delta_x = least_squares(&J, &delta_y, &target.precision);
        let new_x = x + &delta_x;
        new_x
    }
}
