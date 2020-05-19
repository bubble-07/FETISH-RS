extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

pub struct Schmear {
    pub mean : Array1<f32>,
    pub covariance : Array2<f32>
}
