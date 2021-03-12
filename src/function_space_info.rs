extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::feature_space_info::*;
use crate::data_points::*;
use crate::sigma_points::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::alpha_formulas::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::data_point::*;
use rand::prelude::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

#[derive(Clone)]
pub struct FunctionSpaceInfo<'a> {
    pub in_feat_info : &'a FeatureSpaceInfo,
    pub out_feat_info : &'a FeatureSpaceInfo
}

impl <'a> FunctionSpaceInfo<'a> {
    pub fn get_feature_dimensions(&self) -> usize {
        self.in_feat_info.feature_dimensions
    }
    pub fn get_output_dimensions(&self) -> usize {
        self.out_feat_info.get_sketched_dimensions()
    }
    pub fn get_full_dimensions(&self) -> usize {
        self.get_feature_dimensions() * self.get_output_dimensions()
    }

    pub fn jacobian(&self, mat : &Array2<f32>, input : &Array1<f32>) -> Array2<f32> {
        let feat_jacobian = self.in_feat_info.get_feature_jacobian(input);
        let result = mat.dot(&feat_jacobian);
        result
    }
    pub fn apply(&self, mat : &Array2<f32>, input : &Array1<f32>) -> Array1<f32> {
        let features = self.in_feat_info.get_features(input);
        let result = mat.dot(&features);
        result
    }
    pub fn get_data_points(&self, in_data_points : DataPoints) -> DataPoints {
        let feat_vecs = self.in_feat_info.get_features_mat(&in_data_points.in_vecs);
        DataPoints {
            in_vecs : feat_vecs,
            ..in_data_points
        }
    }
    pub fn get_data(&self, in_data : DataPoint) -> DataPoint {
        let feat_vec = self.in_feat_info.get_features(&in_data.in_vec);

        DataPoint {
            in_vec : feat_vec,
            ..in_data
        }
    }
    pub fn apply_schmears(&self, f : &FuncSchmear, x : &Schmear) -> Schmear {
        let feat_schmear = self.in_feat_info.featurize_schmear(x);
        let result = f.apply(&feat_schmear);
        result
    }
}
