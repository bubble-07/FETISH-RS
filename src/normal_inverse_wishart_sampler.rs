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
use crate::wishart::*;
use crate::normal_inverse_wishart::*;

use ndarray_linalg::cholesky::*;

pub struct NormalInverseWishartSampler {
    wishart : Wishart,
    mean : Array2<f32>,
    covariance_cholesky_factor : Array2<f32>,
    t : usize,
    s : usize
}

impl NormalInverseWishartSampler {
    pub fn new(distr : &NormalInverseWishart) -> NormalInverseWishartSampler {
        let wishart : Wishart = Wishart::new(distr.big_v.clone(), distr.little_v);
        let mean = distr.mean.clone();
        let covariance_cholesky_factor = distr.sigma.cholesky(UPLO::Lower).unwrap();
        let s = distr.s;
        let t = distr.t;
        NormalInverseWishartSampler {
            wishart,
            mean,
            covariance_cholesky_factor,
            t,
            s
        }
    }
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let out_chol = self.wishart.sample_cholesky_factor(rng);
        let in_chol = &self.covariance_cholesky_factor;

        let X = Array::random((self.t, self.s), StandardNormal);
	let T = out_chol.dot(&X).dot(in_chol);

        let mut result = self.mean.clone();
        result += &T;
        result
    }
}
