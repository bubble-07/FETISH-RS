extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

///Computes the unique PSD square root of a PSD matrix
pub fn sqrtm(in_mat : &Array2<f32>) -> Array2<f32> {
    let maybe_svd = in_mat.svd(true, true);
    if let Result::Err(_) = &maybe_svd {
        error!("Bad matrix for pseudoinverse {}", in_mat);
    }

    let (maybe_u, sigma, maybe_v_t) = maybe_svd.unwrap();
    let u = maybe_u.unwrap();
    let v_t = maybe_v_t.unwrap();

    let mut sigma_inv = Array::zeros((v_t.shape()[0], u.shape()[1]));
    for i in 0..sigma.shape()[0] {
        let sq_val = sigma[[i,]];
        if (sq_val > 0.0f32) {
            sigma_inv[[i, i]] = sq_val.sqrt(); 
        }
    }
    //Re-constitute for the result
    let result_right = sigma_inv.dot(&u.t().to_owned());
    let result = v_t.t().dot(&result_right);
    result
}


