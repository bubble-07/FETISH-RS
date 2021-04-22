extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::params::*;
use ndarray_linalg::*;

///Yields a matrix mapping from the intrinsic dimension
///of the kernel to the dimension of kernel vectors
///It will be an orthonormal basis by construction
pub fn kernel(in_mat : &Array2<f32>) -> Option<Array2<f32>> {
    let maybe_svd = in_mat.svd(true, true);
    if let Result::Err(_) = &maybe_svd {
        error!("Bad matrix for obtaining kernel {}", in_mat);
    }

    let (_, sigma, maybe_v_t) = maybe_svd.unwrap();
    let v_t = maybe_v_t.unwrap();

    let mut max_singular_value = 0.0f32;
    for i in 0..sigma.shape()[0] {
        max_singular_value = max_singular_value.max(sigma[[i,]]);
    }

    let thresh = max_singular_value * PINV_TRUNCATION_THRESH;

    let mut num_singular_values_above_thresh : usize = 0;
    for i in 0..sigma.shape()[0] {
        if (sigma[[i,]] > thresh) {
            num_singular_values_above_thresh += 1;
        }
    }
    let num_singular_values_below_thresh = in_mat.shape()[1] - num_singular_values_above_thresh;

    if (num_singular_values_below_thresh == 0) {
        Option::None
    } else {
        let mut result = Array::zeros((in_mat.shape()[1], num_singular_values_below_thresh));
        let mut ind : usize = 0;
        for i in num_singular_values_above_thresh..in_mat.shape()[1] {
            let vec = v_t.row(i);
            result.column_mut(ind).assign(&vec);
            ind += 1;
        }
        Option::Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn kernel_is_right_inverse() {
        let matrix = random_matrix(5, 10);
        let kernel = kernel(&matrix).unwrap();
        let vector = random_vector(kernel.shape()[1]);
        let zeros = Array::zeros(matrix.shape()[0]);

        let result = matrix.dot(&kernel).dot(&vector);
        assert_equal_vectors(&result, &zeros);
    }
}
