extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
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

    ///Returns a scalar proportional to the regularization strength
    ///on this feature collection
    fn get_regularization_strength(&self) -> f32;

    ///Yields an empty mean matrix for this feature space in
    ///the specified output dimensionality
    fn blank_mean(&self, out_dims : usize) -> Array2<f32> {
        Array::zeros((out_dims, self.get_dimension()))
    }

    ///Gets the diagonal part of the prior precision matrix
    fn blank_diagonal_precision(&self, out_dims : usize) -> Array4<f32> {
        //Yield the kronecker product
        let scalar = (self.get_dimension() as f32) * self.get_regularization_strength();
        let in_eye = Array::eye(self.get_dimension()) * scalar;
        let out_eye = Array::eye(out_dims);
        einsum("ac,bd->abcd", &[&out_eye, &in_eye])
            .unwrap().into_dimensionality::<Ix4>().unwrap()
    }

    ///Gets the interaction part of the prior precision matrix
    fn blank_interaction_precision(&self, other : &dyn FeatureCollection, out_dims : usize) -> Array4<f32> {
        Array::zeros((out_dims, self.get_dimension(), out_dims, other.get_dimension()))
    }

}
