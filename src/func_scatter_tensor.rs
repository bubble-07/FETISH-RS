extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::linalg_utils::*;
use crate::pseudoinverse::*;

#[derive(Clone)]
pub struct FuncScatterTensor {
    pub in_scatter : Array2<f32>,
    pub out_scatter : Array2<f32>
}

impl FuncScatterTensor {
    pub fn from_in_and_out_scatter(in_scatter : Array2<f32>, out_scatter : Array2<f32>) -> FuncScatterTensor {
        let result = FuncScatterTensor {
            in_scatter,
            out_scatter
        };
        result
    }

    pub fn inverse(&self) -> FuncScatterTensor {
        let in_scatter = pseudoinverse_h(&self.in_scatter);
        let out_scatter = pseudoinverse_h(&self.out_scatter);
        FuncScatterTensor::from_in_and_out_scatter(in_scatter, out_scatter)
    }

    //Supposing that out (A), in (B) scatters are nxn and mxm, respectively,
    //and given a Lx(n*m) matrix C, compute C * kron(A, B)
    //This has better asymptotic efficiency than the naive method
    //Delivers a result which is Lx(n*m)
    pub fn flatten_and_multiply(&self, mat : &Array2<f32>) -> Array2<f32> {
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

    //Flattens and compresses by the given Lx(n*m) projection matrix to yield a LxL covariance
    //matrix
    pub fn compress(&self, mat : &Array2<f32>) -> Array2<f32> {
        let left_transformed = self.flatten_and_multiply(mat);
        let result = left_transformed.dot(&mat.t());
        result
    }

    pub fn flatten(&self) -> Array2<f32> {
        let result = kron(&self.out_scatter, &self.in_scatter);
        result
    }

    ///Transform a t x s mean matrix
    pub fn transform(&self, mean : &Array2<f32>) -> Array2<f32> {
        let mean_in_scatter : Array2<f32> = mean.dot(&self.in_scatter);
        let result = self.out_scatter.dot(&mean_in_scatter);
        result
    }

    ///Induced inner product on t x s mean matrices
    pub fn inner_product(&self, mean_one : &Array2<f32>, mean_two : &Array2<f32>) -> f32 {
        let transformed = self.transform(mean_two);
        let result = frob_inner(mean_one, &transformed);
        result
    }

    pub fn transform_in_out(&self, in_array : &Array2<f32>) -> Array2<f32> {
        let in_inner = frob_inner(&self.in_scatter, in_array);
        let mut out = self.out_scatter.clone();
        out *= in_inner;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_flatten_and_multiply() {
        let in_dim = 5;
        let out_dim = 3;
        let ret_dim = 2;

        let func_scatter_tensor = random_func_scatter_tensor(out_dim, in_dim);
        let mat = random_matrix(ret_dim, out_dim * in_dim);

        let kroned = func_scatter_tensor.flatten();
        let expected = mat.dot(&kroned);

        let actual = func_scatter_tensor.flatten_and_multiply(&mat);

        assert_equal_matrices(&actual, &expected);
    }
}
