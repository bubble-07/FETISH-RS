extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;
use crate::array_utils::*;
use crate::func_scatter_tensor::*;
use crate::pseudoinverse::*;

#[derive(Clone)]
pub struct InverseSchmear {
    pub mean : Array1<f32>,
    pub precision : Array2<f32>
}

impl InverseSchmear {
    pub fn sq_mahalanobis_dist(&self, vec : &Array1<f32>) -> f32 {
        let diff = vec - &self.mean;
        let precision_diff = self.precision.dot(&diff);
        let result : f32 = diff.dot(&precision_diff);
        result
    }
    
    pub fn transform_compress(&self, mat : &Array2<f32>) -> InverseSchmear {
        let mean = mat.dot(&self.mean);
        let full_covariance = pseudoinverse(&self.precision);
        let reduced_covariance = mat.dot(&full_covariance).dot(&mat.t());
        let precision = pseudoinverse(&reduced_covariance);

        InverseSchmear {
            mean,
            precision
        }
    }

    pub fn zero_precision_from_vec(vec : &Array1<f32>) -> InverseSchmear {
        let n = vec.len();
        let precision = Array::zeros((n, n));
        InverseSchmear {
            mean : vec.clone(),
            precision
        }
     }
}
