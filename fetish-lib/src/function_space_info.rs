extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::feature_space_info::*;
use crate::data_points::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::data_point::*;

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

    pub fn jacobian(&self, mat : ArrayView2<f32>, input : ArrayView1<f32>) -> Array2<f32> {
        let feat_jacobian = self.in_feat_info.get_feature_jacobian(input);
        let result = mat.dot(&feat_jacobian);
        result
    }
    pub fn apply(&self, mat : ArrayView2<f32>, input : ArrayView1<f32>) -> Array1<f32> {
        let features = self.in_feat_info.get_features(input);
        let result = mat.dot(&features);
        result
    }
    pub fn get_data_points(&self, in_data_points : DataPoints) -> DataPoints {
        let feat_vecs = self.in_feat_info.get_features_mat(in_data_points.in_vecs.view());
        DataPoints {
            in_vecs : feat_vecs,
            ..in_data_points
        }
    }
    pub fn get_data(&self, in_data : DataPoint) -> DataPoint {
        let feat_vec = self.in_feat_info.get_features(in_data.in_vec.view());

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
