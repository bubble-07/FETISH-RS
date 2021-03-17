extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::sigma_points::*;
use crate::alpha_formulas::*;
use crate::linear_sketch::*;
use crate::enum_feature_collection::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::inverse_schmear::*;

extern crate pretty_env_logger;


#[derive(Clone)]
pub struct FeatureSpaceInfo {
    pub base_dimensions : usize,
    pub feature_dimensions : usize,
    pub feature_collections : Vec<EnumFeatureCollection>,
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
    pub fn sketch(&self, mean : &Array1<f32>) -> Array1<f32> {
        match (&self.sketcher) {
            Option::None => mean.clone(),
            Option::Some(sketch) => sketch.sketch(mean)
        }
    }
    pub fn compress_inverse_schmear(&self, inv_schmear : &InverseSchmear) -> InverseSchmear {
        match (&self.sketcher) {
            Option::None => inv_schmear.clone(),
            Option::Some(sketch) => sketch.compress_inverse_schmear(inv_schmear)
        }
    }
    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        match (&self.sketcher) {
            Option::None => schmear.clone(),
            Option::Some(sketch) => sketch.compress_schmear(schmear)
        }
    }
    pub fn get_feature_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    pub fn get_features_from_base(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        let sketched = self.sketch(in_vec);
        self.get_features(&sketched)
    }

    pub fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    pub fn get_features_mat(&self, in_mat : &Array2<f32>) -> Array2<f32> {
        to_features_mat(&self.feature_collections, in_mat)
    }

    pub fn featurize_schmear(&self, x : &Schmear) -> Schmear {
        let result = unscented_transform_schmear(x, &self); 
        result
    }

    pub fn build_function_feature_space(arg_space : &FeatureSpaceInfo, 
                                        ret_space : &FeatureSpaceInfo) -> FeatureSpaceInfo {
        let base_dimensions = arg_space.feature_dimensions * ret_space.get_sketched_dimensions();
        FeatureSpaceInfo::build_compressed_feature_space(base_dimensions)
    }

    pub fn build_compressed_feature_space(base_dimensions : usize) -> FeatureSpaceInfo {
        let reduced_dimensions = get_reduced_dimension(base_dimensions);
        let alpha = sketch_alpha(base_dimensions);
        let sketcher = Option::Some(LinearSketch::new(base_dimensions, reduced_dimensions, alpha));

        let feature_collections = get_feature_collections(reduced_dimensions);
        let feature_dimensions = get_total_feat_dims(&feature_collections);
        FeatureSpaceInfo {
            base_dimensions,
            feature_dimensions,
            feature_collections,
            sketcher
        }
    }

    pub fn build_uncompressed_feature_space(base_dimensions : usize) -> FeatureSpaceInfo {
        let feature_collections = get_feature_collections(base_dimensions);
        let feature_dimensions = get_total_feat_dims(&feature_collections);
        FeatureSpaceInfo {
            base_dimensions,
            feature_dimensions,
            feature_collections,
            sketcher : Option::None
        }
    }
}
