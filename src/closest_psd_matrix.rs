extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::linalg_utils::*;

pub fn get_closest_unit_norm_psd_matrix(in_mat : &Array2<f32>) -> Array2<f32> {
    let mut result = get_closest_psd_matrix(in_mat);
    let norm = result.opnorm_fro().unwrap();
    result /= norm;
    result
}

pub fn get_closest_psd_matrix(in_mat : &Array2<f32>) -> Array2<f32> {
    let mut symmetrized : Array2<f32> = in_mat.t().clone().to_owned();
    symmetrized += in_mat;
    symmetrized *= 0.5f32;

    let (mut eigenvals, _) = symmetrized.eigh_inplace(UPLO::Lower).unwrap();
    //Threshold the eigenvalues so they're all >= 0
    for i in 0..eigenvals.shape()[0] {
        if (eigenvals[[i,]] < 0.0f32) {
            eigenvals[[i,]] = 0.0f32;
        }
    }
    //Re-constitute the matrix as S^T E S
    let result_right = scale_rows(&symmetrized.t().to_owned(), &eigenvals);
    let result = symmetrized.dot(&result_right);
    result
}
