extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

pub struct DataPoint {
    pub in_vec : Array1<f32>,
    pub out_vec : Array1<f32>,
    pub weight : f32
}

