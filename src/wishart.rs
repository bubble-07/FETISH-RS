extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;

use crate::pseudoinverse::*;
use crate::feature_collection::*;
use crate::params::*;
use crate::test_utils::*;
use crate::linalg_utils::*;

use ndarray_linalg::cholesky::*;

pub struct Wishart {
    pub scale_mat : Array2<f32>,
    pub scale_cholesky_factor : Array2<f32>,
    pub degrees_of_freedom : f32,
    pub dim : usize
}


impl Wishart {
    pub fn new(scale_mat : Array2<f32>, degrees_of_freedom : f32) -> Wishart {
        let scale_cholesky_factor = sqrtm(&scale_mat);
        let dim = scale_mat.shape()[0];
        Wishart {
            scale_mat,
            scale_cholesky_factor,
            degrees_of_freedom,
            dim
        }
    }
    pub fn sample_inv(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let sample = self.sample(rng);
        let result = pseudoinverse_h(&sample);
        result
    }

    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let L = self.sample_cholesky_factor(rng);
	let result = L.dot(&L.t());
        result
    }

    pub fn sample_inv_cholesky_factor(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let sample_inv = self.sample_inv(rng);
        let cholesky_factor = sqrtm(&sample_inv);
        cholesky_factor
    }

    pub fn sample_cholesky_factor(&self, rng : &mut ThreadRng) -> Array2<f32> {
        //Following https://github.com/scipy/scipy/blob/v1.5.1/scipy/stats/_multivariate.py
        //and https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf,
        //first sample a lower-diagonal matrix whose off diagonal elements
        //are random normal variates
        //and whose diagonal elements are chi-square variates

        //Off-diagonal elems
        let mut A = Array::zeros((self.dim, self.dim));
        for i in 0..self.dim {
            for j in 0..i {
                A[[i, j]] = rng.sample(StandardNormal);
            }
        }
        //Diagonal elems
        for i in 0..self.dim {
            let chi_dof = self.degrees_of_freedom - (i as f32);
            let chi = ChiSquared::new(chi_dof).unwrap();            
            let chi_sample = chi.sample(rng);
            let sqrt_chi_sample = chi_sample.sqrt();

            A[[i, i]] = sqrt_chi_sample;
        }

        //Great, now yield the Cholesky factorization of the result
        let result = self.scale_cholesky_factor.dot(&A);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wishart_has_right_mean() {
        let num_samps = 200;
        let dim = 4;
        let degrees_of_freedom = 9;

        let scatter = random_psd_matrix(dim);

        let mut true_mean = scatter.clone();
        true_mean *= degrees_of_freedom as f32;

        let mut actual_mean = Array::zeros((dim,dim));
        let mut rng = rand::thread_rng();

        let wishart = Wishart::new(scatter, degrees_of_freedom as f32);

        for i in 0..num_samps {
            let samp = wishart.sample(&mut rng);
            actual_mean += &samp; 
        }
        let scale_fac = 1.0f32 / (num_samps as f32);
        actual_mean *= scale_fac;

        assert_equal_matrices_to_within(&actual_mean, &true_mean, 16.0f32);
    }
}
