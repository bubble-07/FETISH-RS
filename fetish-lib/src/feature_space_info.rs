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

pub struct FeatureSpaceInfo {
    pub base_dimensions : usize,
    pub feature_dimensions : usize,
    pub feature_collections : Vec<Box<dyn FeatureCollection>>,
    pub sketcher : Option<LinearSketch>
}

impl FeatureSpaceInfo {
    pub fn get_projection_matrix(&self) -> Array2<f32> {
        match (&self.sketcher) {
            Option::None => Array::eye(self.base_dimensions),
            Option::Some(sketch) => sketch.get_projection_matrix().clone()
        }
    }
    pub fn get_sketched_dimensions(&self) -> usize {
        match (&self.sketcher) {
            Option::None => self.base_dimensions,
            Option::Some(sketch) => sketch.get_output_dimension()
        }
    }
    pub fn sketch(&self, mean : ArrayView1<f32>) -> Array1<f32> {
        match (&self.sketcher) {
            Option::None => mean.to_owned(),
            Option::Some(sketch) => sketch.sketch(mean)
        }
    }

    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        match (&self.sketcher) {
            Option::None => schmear.clone(),
            Option::Some(sketch) => sketch.compress_schmear(schmear)
        }
    }
    pub fn get_feature_jacobian(&self, in_vec: ArrayView1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    pub fn get_features_from_base(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        let sketched = self.sketch(in_vec);
        self.get_features(sketched.view())
    }

    pub fn get_features(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    pub fn get_features_mat(&self, in_mat : ArrayView2<f32>) -> Array2<f32> {
        to_features_mat(&self.feature_collections, in_mat)
    }

    pub fn featurize_schmear(&self, x : &Schmear) -> Schmear {
        let result = unscented_transform_schmear(x, &self); 
        result
    }
}
