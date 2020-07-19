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

pub fn outer(a : &Array1<f32>, b : &Array1<f32>) -> Array2<f32> {
    einsum("a,b->ab", &[a, b]).unwrap()
          .into_dimensionality::<Ix2>().unwrap()
}

pub fn frob_inner(a : &Array2<f32>, b : &Array2<f32>) -> f32 {
    einsum("ab,ab->", &[a, b]).unwrap()
          .into_dimensionality::<Ix0>().unwrap().into_scalar()
}
pub fn scale_rows(a : &Array2<f32>, b : &Array1<f32>) -> Array2<f32> {
    einsum("ts,t->ts", &[a, b]).unwrap().into_dimensionality::<Ix2>().unwrap()
}

