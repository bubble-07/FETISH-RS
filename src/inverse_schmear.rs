extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

pub struct InverseSchmear {
    pub mean : Array1<f32>,
    pub precision : Array2<f32>
}
