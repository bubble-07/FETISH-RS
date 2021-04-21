extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::params::*;
use ndarray_linalg::*;

pub fn pseudoinverse_h(in_mat : &Array2<f32>) -> Array2<f32> {
    pseudoinverse(in_mat)
}

pub fn pseudoinverse(in_mat : &Array2<f32>) -> Array2<f32> {
    let maybe_svd = in_mat.svd(true, true);
    if let Result::Err(_) = &maybe_svd {
        error!("Bad matrix for pseudoinverse {}", in_mat);
    }

    let (maybe_u, sigma, maybe_v_t) = maybe_svd.unwrap();
    let u = maybe_u.unwrap();
    let v_t = maybe_v_t.unwrap();

    let mut max_singular_value = 0.0f32;
    for i in 0..sigma.shape()[0] {
        max_singular_value = max_singular_value.max(sigma[[i,]]);
    }
    let thresh = max_singular_value * PINV_TRUNCATION_THRESH;

    let mut sigma_inv = Array::zeros((v_t.shape()[0], u.shape()[1]));
    for i in 0..sigma.shape()[0] {
        if (sigma[[i,]] > thresh) {
            sigma_inv[[i,i]] = 1.0f32 / sigma[[i,]];
        }
    }
    //Re-constitute for the result
    let result_right = sigma_inv.dot(&u.t().to_owned());
    let result = v_t.t().dot(&result_right);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn pseudoinverse_is_inverse_on_square() {
        let matrix = random_matrix(10, 10);
        let matrix_inv = matrix.inv().unwrap();
        let matrix_pinv = pseudoinverse(&matrix);
        assert_equal_matrices(&matrix_pinv, &matrix_inv);
    }

    #[test]
    fn pseudoinverse_of_pseudoinverse_is_identity() {
        let matrix = random_matrix(8, 10);
        let matrix_pinv = pseudoinverse(&matrix);
        let matrix_pinv_pinv = pseudoinverse(&matrix_pinv);
        assert_equal_matrices(&matrix_pinv_pinv, &matrix);
    }

    #[test]
    fn pseudoinverse_has_dims_of_transpose() {
        let matrix = random_matrix(4, 11);
        let matrix_pinv = pseudoinverse(&matrix);
        assert_eq!(matrix_pinv.shape()[0], matrix.shape()[1]);
        assert_eq!(matrix_pinv.shape()[1], matrix.shape()[0]);
    }
}
