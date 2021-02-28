extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use ndarray_linalg::solve::*;
use noisy_float::prelude::*;
use crate::linalg_utils::*;

use crate::inverse_schmear::*;
use crate::pseudoinverse::*;

#[derive(Clone)]
pub struct Schmear {
    pub mean : Array1<f32>,
    pub covariance : Array2<f32>
}

const LN_TWO_PI : f32 = 1.83787706641f32;

impl Schmear {
    pub fn update_lerp(&mut self, mut other : Schmear, lerp_factor : f32) {
        let one_minus_lerp_factor = 1.0f32 - lerp_factor; 
        self.mean *= one_minus_lerp_factor;
        self.covariance *= one_minus_lerp_factor;

        other.mean *= lerp_factor;
        other.covariance *= lerp_factor;

        self.mean += &other.mean;
        self.covariance += &other.covariance; 
    }
    pub fn from_sample_vectors(vecs : &Vec<Array1<f32>>) -> Schmear {
        let d = vecs[0].shape()[0];
        let n = vecs.len();
        let one_over_n = (1.0f32 / (n as f32));
        let one_over_n_minus_one = (1.0f32 / ((n - 1) as f32));

        let mut mean = Array::zeros((d,));
        for vec in vecs.iter() {
            mean += vec;
        }
        mean *= one_over_n;

        let mut covariance = Array::zeros((d, d));
        for vec in vecs.iter() {
            covariance += &outer(vec, vec);
        }
        covariance *= one_over_n_minus_one;
        Schmear {
            mean,
            covariance
        }
    }
    pub fn joint_probability_integral(&self, other : &Schmear) -> f32 {
        let tot_covariance = &self.covariance + &other.covariance;
        let mean_diff = &other.mean - &self.mean;

        let mean_diff_inv = tot_covariance.solveh(&mean_diff).unwrap();
        let sq_mahalanobis_dist = mean_diff.dot(&mean_diff_inv);

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
