extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use crate::rand_utils::*;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;

use crate::sqrtm::*;
use crate::pseudoinverse::*;
use crate::params::*;
use crate::linalg_utils::*;
use crate::ellipsoid::*;

use ndarray_linalg::cholesky::*;

pub struct EllipsoidSampler {
    center : Array1<f32>,
    scatter_cholesky_factor : Array2<f32>
}

impl EllipsoidSampler {
    pub fn new(ellipsoid : &Ellipsoid) -> EllipsoidSampler {
        let skew = ellipsoid.skew();
        let scatter = pseudoinverse_h(skew);
        let scatter_cholesky_factor = sqrtm(&scatter);
        let center = ellipsoid.center().clone();
        EllipsoidSampler {
            center,
            scatter_cholesky_factor
        }
    }
    pub fn sample(&self, rng : &mut ThreadRng) -> Array1<f32> {
        let unit_ball_vec = gen_nball_random(rng, self.center.shape()[0]);
        let mut skewed_vec = self.scatter_cholesky_factor.dot(&unit_ball_vec);
        skewed_vec += &self.center;
        skewed_vec
    }
}
