extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;
use crate::array_utils::*;
use crate::func_scatter_tensor::*;

#[derive(Clone)]
pub struct InverseSchmear {
    pub mean : Array1<f32>,
    pub precision : Array2<f32>
}

impl InverseSchmear {
    pub fn mahalanobis_dist(&self, vec : &Array1<f32>) -> f32 {
        let diff = vec - &self.mean;
        let precision_diff = self.precision.dot(&diff);
        let result : f32 = diff.dot(&precision_diff);
        result
    }
    
    pub fn transform_compress(&self, mat : &Array2<f32>, mat_pinv : &Array2<f32>) -> InverseSchmear {
        let mean = mat.dot(&self.mean);
        let precision = mat_pinv.t().dot(&self.precision).dot(mat_pinv);
        InverseSchmear {
            mean,
            precision
        }
    }

    pub fn ident_precision_from_noisy(vec : &Array1<R32>) -> InverseSchmear {
        let n = vec.len();
        let precision = Array::eye(n);
        let mean = from_noisy(vec);
        InverseSchmear {
            mean,
            precision
        }
    }
}
