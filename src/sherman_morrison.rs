extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::linalg_utils::*;
use std::ops::AddAssign;
use std::ops::SubAssign;

//Compute A + w * uu^T and its inverse in one stroke
//for the case where A is already symmetric
pub fn sherman_morrison_update(A : &mut Array2<f32>, A_inv : &mut Array2<f32>,
                        w : f32, u : &Array1<f32>) {
    let w_u_outer_u = w * outer(u, u);
    A.add_assign(&w_u_outer_u);

    let A_inv_u = A_inv.dot(u);
    let numerator = w * outer(&A_inv_u, &A_inv_u);

    let w_u_A_inv_u = w * u.dot(&A_inv_u);
    let denominator = 1.0f32 + w_u_A_inv_u;

    let scale = 1.0f32 / denominator;
    let scaled = scale * &numerator;

    A_inv.sub_assign(&scaled);
}
