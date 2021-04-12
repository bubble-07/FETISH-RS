extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::linalg_utils::*;
use std::ops::AddAssign;
use std::ops::SubAssign;

//Compute A + w * uu^T and its inverse in one stroke
//for the case where A is already symmetric
pub fn sherman_morrison_update(A : &mut Array2<f32>, A_inv : &mut Array2<f32>,
                        w : f32, u : ArrayView1<f32>) {
    let w_u_outer_u = w * outer(u, u);
    A.add_assign(&w_u_outer_u);

    let A_inv_u = A_inv.dot(&u);
    let numerator = w * outer(A_inv_u.view(), A_inv_u.view());

    let w_u_A_inv_u = w * u.dot(&A_inv_u);
    let denominator = 1.0f32 + w_u_A_inv_u;

    let scale = 1.0f32 / denominator;
    let scaled = scale * &numerator;

    A_inv.sub_assign(&scaled);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pseudoinverse::*;
    use crate::test_utils::*;

    #[test]
    fn test_sherman_morrison() {
        let dim = 5;
        let A = random_psd_matrix(dim);
        let A_inv = pseudoinverse_h(&A);
        let w = random_scalar();
        let u = random_vector(dim); 

        let mut A_updated = A.clone(); 
        let mut A_inv_updated = A_inv.clone();
        sherman_morrison_update(&mut A_updated, &mut A_inv_updated, w, &u);

        let expected_A_inv = pseudoinverse_h(&A_updated);

        assert_equal_matrices_to_within(&A_inv_updated, &expected_A_inv, 0.001f32);
    }
}
