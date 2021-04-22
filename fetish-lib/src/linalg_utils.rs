extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops::MulAssign;

///Computes the squared Euclidean distance between two vectors
pub fn sq_vec_dist(one : ArrayView1<f32>, two : ArrayView1<f32>) -> f32 {
    let diff = &one - &two;
    diff.dot(&diff)
}

///Computes the outer product `ab^T` of vectors `a` and `b`.
pub fn outer(a : ArrayView1<f32>, b : ArrayView1<f32>) -> Array2<f32> {
    let a_column = into_col(a.clone());
    let b_row = into_row(b.clone());
    a_column.dot(&b_row)
}

///Computes the Frobenius inner product of two matrices, which
///is the same as computing the dot product of the vectorized matrices.
pub fn frob_inner(a : ArrayView2<f32>, b : ArrayView2<f32>) -> f32 {
    let flat_a = flatten(a.clone());
    let flat_b = flatten(b.clone());
    flat_a.dot(&flat_b)
}

///Scales the rows of `a` by the respective scaling factors in `b`. Useful
///for efficiently computing left-multiplication by a diagonal matrix.
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
///Computes the Kronecker product of matrices.
pub fn kron(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    //Pulled from: https://github.com/rust-ndarray/ndarray/issues/652
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * &b));
    }
    out
}

