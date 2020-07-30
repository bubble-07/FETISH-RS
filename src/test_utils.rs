extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::params::*;
use ndarray_linalg::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use crate::linalg_utils::*;

pub fn assert_equal_schmears(one : &Schmear, two : &Schmear) {
    assert_equal_matrices(&one.covariance, &two.covariance);
    assert_equal_vectors(&one.mean, &two.mean);
}

pub fn assert_equal_inv_schmears(one : &InverseSchmear, two : &InverseSchmear) {
    assert_equal_matrices(&one.precision, &two.precision);
    assert_equal_vectors(&one.mean, &two.mean);
}

pub fn assert_equal_matrices(one : &Array2<f32>, two : &Array2<f32>) {
    let diff = one - two;
    let frob_norm = diff.opnorm_fro().unwrap();
    if (frob_norm > ZEROING_THRESH) {
        panic!();
    }
}
pub fn assert_equal_vectors(one : &Array1<f32>, two : &Array1<f32>) {
    let diff = one - two;
    let sq_norm = diff.dot(&diff);
    let norm = sq_norm.sqrt();
    if (norm > ZEROING_THRESH) {
        panic!();
    }
}
pub fn assert_eps_equals(one : f32, two : f32) {
    let diff = one - two;
    if (diff.abs() > ZEROING_THRESH) {
        println!("Actual: {} Expected: {}", one, two);
        panic!();
    }
}
pub fn random_vector(t : usize) -> Array1<f32> {
    Array::random((t,), StandardNormal)
}
pub fn random_matrix(t : usize, s : usize) -> Array2<f32> {
    Array::random((t, s), StandardNormal)
}
pub fn random_diag_matrix(t : usize) -> Array2<f32> {
    let mut result = Array::zeros((t, t));
    let diag = Array::random((t,), StandardNormal);
    for i in 0..t {
        result[[i, i]] = diag[[i,]];
    }
    result
}
pub fn random_psd_matrix(t : usize) -> Array2<f32> {
    let matrix_sqrt = random_diag_matrix(t);
    let matrix = matrix_sqrt.t().dot(&matrix_sqrt);
    matrix
}
pub fn random_schmear(t : usize) -> Schmear {
    let covariance = random_psd_matrix(t);
    let mean = random_vector(t);
    Schmear {
        mean,
        covariance
    }
}
pub fn random_inv_schmear(t : usize) -> InverseSchmear {
    let precision = random_psd_matrix(t);
    let mean = random_vector(t);
    InverseSchmear {
        mean,
        precision
    }
}
