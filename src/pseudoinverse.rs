extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::linalg_utils::*;

pub fn pseudoinverse(in_mat : &Array2<f32>) -> Array2<f32> {
    let (maybe_u, sigma, maybe_v_t) = in_mat.svd(true, true).unwrap();
    let u = maybe_u.unwrap();
    let v_t = maybe_v_t.unwrap();

    let mut sigma_inv = Array::zeros((v_t.shape()[0], u.shape()[1]));
    for i in 0..sigma.shape()[0] {
        if (sigma[[i,]] > 0.0f32) {
            sigma_inv[[i,i]] = 1.0f32 / sigma[[i,]];
        }
    }
    //Re-constitute for the result
    let result_right = sigma_inv.dot(&u.t().to_owned());
    let result = v_t.t().dot(&result_right);
    result
}
