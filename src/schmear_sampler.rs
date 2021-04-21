extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use fetish_lib::everything::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;


pub struct SchmearSampler {
    center : Array1<f32>,
    covariance_cholesky_factor : Array2<f32>
}

impl SchmearSampler {
    pub fn new(schmear : &Schmear) -> SchmearSampler {
        let covariance_cholesky_factor = sqrtm(&schmear.covariance);
        let center = schmear.mean.clone();
        SchmearSampler {
            center,
            covariance_cholesky_factor
        }
    }
    pub fn sample(&self, rng : &mut ThreadRng) -> Array1<f32> {
        let n = self.center.shape()[0];
        let std_norm_vec = Array::random_using((n,), StandardNormal, rng);
        let mut result = self.covariance_cholesky_factor.dot(&std_norm_vec);
        result += &self.center;
        result
    }
}
