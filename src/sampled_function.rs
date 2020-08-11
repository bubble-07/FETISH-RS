extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::rc::*;
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
    pub feature_collections : Rc<[EnumFeatureCollection; 3]>
}

impl SampledFunction {
    pub fn apply(&self, input : &Array1<f32>) -> Array1<f32> {
        let features : Array1<f32> = to_features(&self.feature_collections, input);
        let result : Array1<f32> = self.mat.dot(&features);
        result
    }
}
