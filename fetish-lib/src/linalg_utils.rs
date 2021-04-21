extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops::MulAssign;

pub fn sq_vec_dist(one : ArrayView1<f32>, two : ArrayView1<f32>) -> f32 {
    let diff = &one - &two;
    diff.dot(&diff)
}

pub fn outer(a : ArrayView1<f32>, b : ArrayView1<f32>) -> Array2<f32> {
    let a_column = into_col(a.clone());
    let b_row = into_row(b.clone());
    a_column.dot(&b_row)
}

pub fn normalize_frob(a : ArrayView2<f32>) -> Array2<f32> {
    let sq_norm = frob_inner(a, a);
    (1.0f32 / sq_norm) * a.to_owned()
}

pub fn frob_inner(a : ArrayView2<f32>, b : ArrayView2<f32>) -> f32 {
    let flat_a = flatten(a.clone());
    let flat_b = flatten(b.clone());
    flat_a.dot(&flat_b)
}
pub fn scale_rows(a : ArrayView2<f32>, b : ArrayView1<f32>) -> Array2<f32> {
    let mut result = a.to_owned();
    let n = a.shape()[0];
    for i in 0..n {
        let scale = b[[i,]];
        let mut row = result.row_mut(i);
        row.mul_assign(scale);
    }
    result
}
//Pulled from: https://github.com/rust-ndarray/ndarray/issues/652
pub fn kron(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * &b));
    }
    out
}

