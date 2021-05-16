extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::linalg_utils::*;
use crate::pseudoinverse::*;

///Represents a 4-tensor which is expressible as
///`kron(out_scatter, in_scatter)`, where `out_scatter`
///and `in_scatter` are both positive semi-definite.
///This is typically used to express the precision
///or covariance of a [`crate::normal_inverse_wishart::NormalInverseWishart`] model.
///See also [`crate::func_schmear::FuncSchmear`] for an example of usage.
#[derive(Clone)]
pub struct FuncScatterTensor {
    pub in_scatter : Array2<f32>,
    pub out_scatter : Array2<f32>
}

impl FuncScatterTensor {
    ///Inverts both `in_scatter` and `out_scatter`
    ///of this [`FuncScatterTensor`] to yield
    ///a new one. May be used to e.g: convert between
    ///a function's covariance 4-tensor and its precision 4-tensor.
    pub fn inverse(&self) -> FuncScatterTensor {
        let in_scatter = pseudoinverse_h(&self.in_scatter);
        let out_scatter = pseudoinverse_h(&self.out_scatter);
        FuncScatterTensor {
            in_scatter,
            out_scatter
        }
    }

    ///Supposing that `out_scatter` and `in_scatter` are `nxn` and `mxm`, respectively,
    ///and given a `Lx(n*m)` matrix `mat`, compute `mat * kron(out_scatter, in_scatter)`. 
    ///This has better asymptotic efficiency than simply expanding out the definition
    ///above. Delivers a result of size `Lx(n*m)`. 
    pub fn flatten_and_multiply(&self, mat : ArrayView2<f32>) -> Array2<f32> {
        let out_dim = self.out_scatter.shape()[0];
        let in_dim = self.in_scatter.shape()[0];
        let ret_dim = mat.shape()[0];

        let in_exposed_mat = mat.clone().into_shape((ret_dim * out_dim, in_dim)).unwrap();
        let in_transformed_mat = in_exposed_mat.dot(&self.in_scatter);

        let mut in_transformed_tensor = in_transformed_mat.into_shape((ret_dim, out_dim, in_dim)).unwrap();
        in_transformed_tensor.swap_axes(1, 2);
        let in_transformed_tensor = in_transformed_tensor.as_standard_layout();

        let out_exposed_mat = in_transformed_tensor.into_shape((ret_dim * in_dim, out_dim)).unwrap();
        let out_transformed_mat = out_exposed_mat.dot(&self.out_scatter);

        let mut out_transformed_tensor = out_transformed_mat.into_shape((ret_dim, in_dim, out_dim)).unwrap();
        out_transformed_tensor.swap_axes(1, 2);
        let out_transformed_tensor = out_transformed_tensor.as_standard_layout();

        let result = out_transformed_tensor.into_shape((ret_dim, out_dim * in_dim)).unwrap();
        result.into_owned()
    }

    ///Efficiently transforms this [`FuncScatterTensor`], taken to represent a covariance tensor,
    ///through the linear transformation given by `mat`.
    ///
    ///Supposing that `out_scatter` and `in_scatter` are `nxn` and `mxm`, respectively,
    ///and given a `Lx(n*m)` matrix `mat`, compute `mat * kron(out_scatter, in_scatter) * mat.t()`.
    ///This has better asymptotic efficiency than simply expanding out the definition above.
    ///Delivers a result of size `LxL`.
    pub fn compress(&self, mat : ArrayView2<f32>) -> Array2<f32> {
        let left_transformed = self.flatten_and_multiply(mat);
        let result = left_transformed.dot(&mat.t());
        result
    }

    ///Flattens this [`FuncScatterTensor`] to a 2d matrix using the kronecker product
    ///`kron(out_scatter, in_scatter)`.
    pub fn flatten(&self) -> Array2<f32> {
        let result = kron(self.out_scatter.view(), self.in_scatter.view());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_flatten_and_multiply() {
        let in_dim = 5;
        let out_dim = 3;
        let ret_dim = 2;

        let func_scatter_tensor = random_func_scatter_tensor(out_dim, in_dim);
        let mat = random_matrix(ret_dim, out_dim * in_dim);

        let kroned = func_scatter_tensor.flatten();
        let expected = mat.dot(&kroned);

        let actual = func_scatter_tensor.flatten_and_multiply(mat.view());

        assert_equal_matrices(actual.view(), expected.view());
    }
}
