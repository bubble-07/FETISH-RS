extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use enum_dispatch::*;

#[enum_dispatch]
pub trait FeatureCollection {
    ///Return the number of input dimensions
    fn get_in_dimensions(&self) -> usize;

    ///Return the number of dimensions in the output of get_features
    fn get_dimension(&self) -> usize;

    ///Given a vector in the input space of this feature space, return the
    ///vector of features for this feature space
    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32>;

    ///Given a vector in the input space of this feature space, return
    ///the Jacobian matrix for the feature vector at that point
    ///in the format f x s, for s the input space size
    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32>;

    fn get_features_mat(&self, in_mat : &Array2<f32>) -> Array2<f32> {
        let n = in_mat.shape()[0];
        let d = self.get_dimension();
        let mut result = Array::zeros((n, d));
        for i in 0..n {
            let in_vec = in_mat.row(i).to_owned();
            let feat_vec = self.get_features(&in_vec);
            result.row_mut(i).assign(&feat_vec);
        }
        result
    }
}
