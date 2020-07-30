extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::linalg_utils::*;
use crate::test_utils::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_psd_of_diagonal_is_diagonal() {
        let matrix_sqrt = random_diag_matrix(10);
        let matrix = matrix_sqrt.t().dot(&matrix_sqrt);
        let closest_psd_matrix = get_closest_psd_matrix(&matrix);
        assert_equal_matrices(&closest_psd_matrix, &matrix);
    }

    #[test]
    fn closest_psd_of_psd_is_identity() {
        let matrix = random_psd_matrix(10);
        let closest_psd_matrix = get_closest_psd_matrix(&matrix);
        assert_equal_matrices(&closest_psd_matrix, &matrix); 
    }

}
