extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;
use crate::params::*;

//Note: This draws from https://docs.rs/crate/petal-decomposition/0.4.0/source/src/pca.rs

pub fn randomized_rank_one_approx(input : &Array2<f32>, rng : &mut ThreadRng) -> (Array1<f32>, Array1<f32>) {
    let t = input.shape()[0];
    let s = input.shape()[1];
    let (u, sigma, vt) = randomized_svd(input, 1, rng);
    let u_flat = u.into_shape((t,)).unwrap();
    let mut v_flat = vt.into_shape((s,)).unwrap();
    v_flat *= sigma[[0,]];
    (u_flat, v_flat)
}

fn randomized_svd(input : &Array2<f32>, n_components : usize, rng : &mut ThreadRng) -> 
                                            (Array2<f32>, Array1<f32>, Array2<f32>) {
    let n_random = n_components + SVD_OVERSAMPLE; //oversample by 10 to find the range
    let q = randomized_range_finder(input, n_random, SVD_RANGE_ITERS, rng);
    let b = q.t().dot(input);
    let (u, sigma, vt) = b.svddc(UVTFlag::Some).unwrap();
    let mut u = q.dot(&u.unwrap());
    (u, sigma, vt.unwrap())
}

fn randomized_range_finder(input : &Array2<f32>, size : usize, n_iter : usize, rng : &mut ThreadRng) -> 
                                                                               Array2<f32> {
    let input_t = input.t();
    //This draws from algorithm 4.4 of https://arxiv.org/pdf/0909.4061.pdf
    let mut omega = Array::random((input.ncols(), size), StandardNormal); 

    let mut y = input.dot(&omega);
    let (mut q, _) = y.qr().unwrap();
    for _ in 0..n_iter {
        let y_twiddle = input_t.dot(&q);
        let (q_twiddle, _) = y_twiddle.qr().unwrap();
        let y = input.dot(&q_twiddle);
        let (q, _) = y.qr().unwrap();
    }
    q
}
