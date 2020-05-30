extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

use crate::inverse_schmear::*;

pub struct Schmear {
    pub mean : Array1<f32>,
    pub covariance : Array2<f32>
}

impl Schmear {
    pub fn inverse(&self) -> InverseSchmear {
        InverseSchmear {
            mean : self.mean.clone(),
            precision : self.covariance.clone().invh().unwrap()
        }
    }
}
