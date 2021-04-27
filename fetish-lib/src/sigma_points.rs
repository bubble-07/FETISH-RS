extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::schmear::*;
use crate::feature_space_info::*;
use crate::sqrtm::*;

///Gets the canonical set of 2n + 1 sigma points (see: unscented transform)
///for the given [`Schmear`]. These points are chosen in such a way
///that their mean and covariance match that of the given schmear.
pub fn get_sigma_points(in_schmear : &Schmear) -> Vec<Array1<f32>> {
    let mean = &in_schmear.mean;
    let n = mean.shape()[0];

    let mut covariance_sqrt = sqrtm(&in_schmear.covariance);
    let n_sqrt = (n as f32).sqrt();  
    covariance_sqrt *= n_sqrt;

    let mut result = Vec::new();
    result.push(mean.clone());
    for i in 0..n {
        let covariance_col = covariance_sqrt.column(i);
        let plus_vec = mean + &covariance_col;
        let minus_vec = mean - &covariance_col;
        result.push(plus_vec);
        result.push(minus_vec);
    }

    result
}

///Given some points, yields a [`Schmear`] with their mean and (sample) covariance.
pub fn sigma_points_to_schmear(in_points : &Vec<Array1<f32>>) -> Schmear {
    Schmear::from_sample_vectors(in_points)
}

///Given a [`Schmear`] in a compressed space and some [`FeatureSpaceInfo`], computes
///an estimated featurized [`Schmear`] using an unscented transform on the canonical set
///of 2n + 1 sigma points.
pub fn unscented_transform_schmear(in_schmear : &Schmear, feat_space_info : &FeatureSpaceInfo) -> Schmear {
    let in_sigma_points = get_sigma_points(in_schmear);
    let mut out_sigma_points = Vec::new();
    for in_sigma_point in in_sigma_points {
        let out_sigma_point = feat_space_info.get_features(in_sigma_point.view());
        out_sigma_points.push(out_sigma_point);
    }
    
    let result = sigma_points_to_schmear(&out_sigma_points);
    result
}
