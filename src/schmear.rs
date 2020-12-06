extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;

use crate::inverse_schmear::*;
use crate::pseudoinverse::*;

pub struct Schmear {
    pub mean : Array1<f32>,
    pub covariance : Array2<f32>
}

impl Schmear {
    pub fn from_vector(vec : &Array1<R32>) -> Schmear {
        let n = vec.len();
        let mean = from_noisy(vec);
        let covariance : Array2::<f32> = Array::zeros((n, n));
        Schmear {
            mean : mean,
            covariance : covariance
        }
    }
    pub fn inverse(&self) -> InverseSchmear {
        let mean = self.mean.clone();
        let precision = pseudoinverse(&self.covariance);
        InverseSchmear {
            mean,
            precision
        }
    }
    pub fn transform_compress(&self, mat : &Array2<f32>) -> Schmear {
        let mean = mat.dot(&self.mean);
        let covariance = mat.dot(&self.covariance).dot(&mat.t());
        Schmear {
            mean,
            covariance
        }
    }
}
