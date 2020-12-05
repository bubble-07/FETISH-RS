extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::data_points::*;
use crate::sigma_points::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::alpha_formulas::*;
use crate::vector_space::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::data_point::*;
use rand::prelude::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

#[derive(Clone)]
pub struct SpaceInfo {
    pub in_dimensions : usize,
    pub feature_dimensions : usize,
    pub out_dimensions : usize,
    pub feature_collections : Vec<EnumFeatureCollection>,
    pub func_sketcher : LinearSketch
}

impl SpaceInfo {
    pub fn get_full_dimensions(&self) -> usize {
        self.feature_dimensions * self.out_dimensions
    }
    pub fn get_sketched_dimensions(&self) -> usize {
        self.func_sketcher.get_output_dimension()
    }
    pub fn sketch(&self, mean : &Array1<f32>) -> Array1<f32> {
        self.func_sketcher.sketch(mean)
    }
    pub fn compress_inverse_schmear(&self, inv_schmear : &InverseSchmear) -> InverseSchmear {
        self.func_sketcher.compress_inverse_schmear(inv_schmear)
    }
    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        self.func_sketcher.compress_schmear(schmear)
    }

    pub fn inflate_compressed_vector(&self, compressed_func : &Array1<f32>) -> Array2<f32> {
        let full_flat_func = self.func_sketcher.expand(compressed_func);
        let result = full_flat_func.into_shape((self.out_dimensions, self.feature_dimensions)).unwrap();
        result
    }

    pub fn jacobian(&self, mat : &Array2<f32>, input : &Array1<f32>) -> Array2<f32> {
        let feat_jacobian = to_jacobian(&self.feature_collections, input);
        let result : Array2<f32> = mat.dot(&feat_jacobian);
        result
    }

    pub fn apply(&self, mat : &Array2<f32>, input : &Array1<f32>) -> Array1<f32> {
        let features : Array1<f32> = to_features(&self.feature_collections, input);
        let result : Array1<f32> = mat.dot(&features);
        result
    }
    

    pub fn get_feature_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    pub fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    pub fn get_features_mat(&self, in_mat : &Array2<f32>) -> Array2<f32> {
        to_features_mat(&self.feature_collections, in_mat)
    }

    pub fn get_data_points(&self, in_data_points : DataPoints) -> DataPoints {
        let feat_vecs = self.get_features_mat(&in_data_points.in_vecs);
        DataPoints {
            in_vecs : feat_vecs,
            ..in_data_points
        }
    }

    pub fn get_data(&self, in_data : DataPoint) -> DataPoint {
        let feat_vec = self.get_features(&in_data.in_vec);

        DataPoint {
            in_vec : feat_vec,
            ..in_data
        }
    }

    fn featurize_schmear(&self, x : &Schmear) -> Schmear {
        let result = unscented_transform_schmear(x, &self); 
        result
    }

    pub fn apply_schmears(&self, f : &FuncSchmear, x : &Schmear) -> Schmear {
        let feat_schmear = self.featurize_schmear(x);
        let result = f.apply(&feat_schmear);
        result
    }

    pub fn new(in_dimensions : usize, out_dimensions : usize) -> SpaceInfo {
        let feature_collections = get_feature_collections(in_dimensions);
        let total_feat_dims = get_total_feat_dims(&feature_collections);

        info!("And feature dims {}", total_feat_dims);

        let embedding_dim = total_feat_dims * out_dimensions;
        let sketched_embedding_dim = get_reduced_output_dimension(embedding_dim);
        let alpha = sketch_alpha(embedding_dim);

        let output_sketch = LinearSketch::new(embedding_dim, sketched_embedding_dim, alpha);

        let result = SpaceInfo {
            in_dimensions : in_dimensions,
            feature_dimensions : total_feat_dims,
            out_dimensions : out_dimensions,
            feature_collections : feature_collections,
            func_sketcher : output_sketch
        };
        result
    }

}
