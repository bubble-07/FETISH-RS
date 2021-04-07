extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::fourier_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::sketched_linear_feature_collection::*;
use crate::rand_utils::*;

pub trait FeatureCollection : Sync {
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

pub fn get_feature_collections(in_dimensions : usize) -> Vec<Box<dyn FeatureCollection>> {
    let linear_collection = SketchedLinearFeatureCollection::new(in_dimensions);
    let quadratic_collection = QuadraticFeatureCollection::new(in_dimensions);
    let fourier_collection = FourierFeatureCollection::new(in_dimensions, gen_cauchy_random);

    let mut result = Vec::<Box<dyn FeatureCollection>>::new();
    result.push(Box::new(linear_collection));
    result.push(Box::new(quadratic_collection));
    result.push(Box::new(fourier_collection));

    result
}

pub fn get_total_feat_dims(feature_collections : &Vec<Box<dyn FeatureCollection>>) -> usize {
    let mut total_feat_dims : usize = 0;
    for collection in feature_collections.iter() {
        total_feat_dims += collection.get_dimension();
    }
    total_feat_dims
}
