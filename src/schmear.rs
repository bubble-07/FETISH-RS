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

const LN_TWO_PI : f32 = 1.83787706641f32;

impl Schmear {
    pub fn joint_probability_integral(&self, other : &Schmear) -> f32 {
        let tot_covariance = &self.covariance + &other.covariance;
        let tot_covariance_inv = pseudoinverse_h(&tot_covariance);
        let mean_diff = &other.mean - &self.mean;
        let sq_mahalanobis_dist = mean_diff.dot(&tot_covariance_inv).dot(&mean_diff);

        let maybe_tot_covariance_sln_det = tot_covariance.sln_deth();
        let (_, tot_covariance_ln_det) = maybe_tot_covariance_sln_det.unwrap(); 
         
        let total_exponent = -0.5f32 * (LN_TWO_PI + tot_covariance_ln_det + sq_mahalanobis_dist);
        let result = total_exponent.exp();
        result
    }
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
