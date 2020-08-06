extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::eigh::*;
use ndarray_linalg::lapack::*;

pub fn sq_vec_dist(one : &Array1<f32>, two : &Array1<f32>) -> f32 {
    let diff = one - two;
    diff.dot(&diff)
}

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
//Pulled from: https://github.com/rust-ndarray/ndarray/issues/652
pub fn kron(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * b));
    }
    out
}

