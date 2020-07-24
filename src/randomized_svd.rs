extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::lobpcg::TruncatedSvd;
use ndarray_linalg::lobpcg::TruncatedOrder;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;
use crate::params::*;

pub fn randomized_rank_one_approx(input : &Array2<f32>, rng : &mut ThreadRng) -> (Array1<f32>, f32, Array1<f32>) {
    let t = input.shape()[0];
    let s = input.shape()[1];

    let mut svd = TruncatedSvd::new(input.clone(), TruncatedOrder::Largest);
    svd = svd.precision(SVD_PRECISION);
    svd = svd.maxiter(SVD_MAX_ITER);

    let svd_result = svd.decompose(1).unwrap();
    let (u, sigma, vt) = svd_result.values_vectors();

    let u_flat = u.into_shape((t,)).unwrap();
    let v_flat = vt.into_shape((s,)).unwrap();
    (u_flat, sigma[[0,]], v_flat)
}
