extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::sigma_points::*;
use crate::linear_sketch::*;
use crate::feature_collection::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::inverse_schmear::*;

///Information about the base space dimension, the preferred
///sketcher for the space, the feature dimensions, and the feature mapping.
///In other words, this struct carries all information defining spaces which
///are associated to a particular type.
///
///As in the FETISH paper, the way this is conceptually organized is:
///Base space -(sketcher)-> Compressed space -(feature mapping)-> Feature space.
pub struct FeatureSpaceInfo {
    pub base_dimensions : usize,
    pub feature_dimensions : usize,
    pub feature_collections : Vec<Box<dyn FeatureCollection>>,
    pub sketcher : Option<LinearSketch>
}

impl FeatureSpaceInfo {
    ///Gets the projection matrix from the base space to the compressed space.
    pub fn get_projection_matrix(&self) -> Array2<f32> {
        match (&self.sketcher) {
            Option::None => Array::eye(self.base_dimensions),
            Option::Some(sketch) => sketch.get_projection_matrix().clone()
        }
    }
    ///Gets the dimensionality of the compressed space.
    pub fn get_sketched_dimensions(&self) -> usize {
        match (&self.sketcher) {
            Option::None => self.base_dimensions,
            Option::Some(sketch) => sketch.get_output_dimension()
        }
    }
    ///Transforms a vector from the base space to the compressed space.
    pub fn sketch(&self, mean : ArrayView1<f32>) -> Array1<f32> {
        match (&self.sketcher) {
            Option::None => mean.to_owned(),
            Option::Some(sketch) => sketch.sketch(mean)
        }
    }

    ///Transforms a schmear from the base space to the compressed space.
    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        match (&self.sketcher) {
            Option::None => schmear.clone(),
            Option::Some(sketch) => sketch.compress_schmear(schmear)
        }
    }
    ///Gets the Jacobian for the feature mapping evaluated at the given compressed vector.
    pub fn get_feature_jacobian(&self, in_vec: ArrayView1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    ///Maps a vector from the base space through the sketcher and through the feature map
    ///to yield a vector in the feature space.
    pub fn get_features_from_base(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        let sketched = self.sketch(in_vec);
        self.get_features(sketched.view())
    }

    ///Gets the feature vector for the given compressed vector.
    pub fn get_features(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    ///See [`to_features_mat`]. Gets features for a collection of compressed vectors.
    pub fn get_features_mat(&self, in_mat : ArrayView2<f32>) -> Array2<f32> {
        to_features_mat(&self.feature_collections, in_mat)
    }

    ///Given a [`Schmear`] in the compressed space, uses an unscented transform to
    ///estimate what it would map to through the feature mapping.
    pub fn featurize_schmear(&self, x : &Schmear) -> Schmear {
        let result = unscented_transform_schmear(x, &self); 
        result
    }
}
