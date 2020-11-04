extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use noisy_float::prelude::*;

pub fn from_noisy(vec : &Array1<R32>) -> Array1<f32> {
    let n = vec.len();
    let mut mean : Array1::<f32> = Array::zeros((n,));
    for i in 0..n {
        mean[[i,]] = vec[[i,]].raw();
    }
    mean
}

pub fn to_noisy(vec : &Array1<f32>) -> Array1<R32> {
    let n = vec.len();
    let mut result : Array1::<R32> = Array::zeros((n,));
    for i in 0..n {
        result[[i,]] = r32(vec[[i,]]);
    }
    result
}

pub fn max_index_and_value(vec : &ArrayView1<f32>) -> (usize, f32) {
    let mut max_index = 0; 
    let mut max_value = vec[[0,]];
    for i in 1..vec.shape()[0] {
        if (vec[[i,]] > max_value) {
            max_value = vec[[i,]];
            max_index = i;
        }
    }
    (max_index, max_value)
}
