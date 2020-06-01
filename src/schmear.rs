extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;

use crate::inverse_schmear::*;

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

    fn inverse(&self) -> InverseSchmear {
        InverseSchmear {
            mean : self.mean.clone(),
            precision : self.covariance.clone().invh().unwrap()
        }
    }
}
