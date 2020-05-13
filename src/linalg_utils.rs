extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::eigh::*;
use ndarray_linalg::lapack::*;

pub fn sqrtm(in_array: &Array2<f32>) -> Array2<f32> {
    in_array.ssqrt(UPLO::Lower).unwrap()
}
