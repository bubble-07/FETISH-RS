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
