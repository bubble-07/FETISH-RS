extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;

use crate::feature_collection::*;
use crate::params::*;

use ndarray_linalg::cholesky::*;

pub struct Wishart {
    pub scale_mat : Array2<f32>,
    pub scale_cholesky_factor : Array2<f32>,
    pub degrees_of_freedom : f32,
    pub dim : usize
}


impl Wishart {
    pub fn new(scale_mat : Array2<f32>, degrees_of_freedom : f32) -> Wishart {
        let scale_cholesky_factor = scale_mat.cholesky(UPLO::Lower).unwrap();
        let dim = scale_mat.shape()[0];
        Wishart {
            scale_mat,
            scale_cholesky_factor,
            degrees_of_freedom,
            dim
        }
    }
    pub fn sample_inv(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.sample(rng).invh().unwrap()
    }

    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let L = self.sample_cholesky_factor(rng);
	let result = L.dot(&L.t());
        result
    }

    pub fn sample_inv_cholesky_factor(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.sample_cholesky_factor(rng).inv().unwrap()
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
