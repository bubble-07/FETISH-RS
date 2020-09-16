extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

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
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::sampled_function::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use arraymap::ArrayMap;
use rand::prelude::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

type ModelKey = usize;

pub struct ModelSpace {
    pub in_dimensions : usize,
    pub feature_dimensions : usize,
    pub out_dimensions : usize,
    pub feature_collections : Rc<[EnumFeatureCollection; 3]>,
    models : HashMap<ModelKey, Model>,
    func_sketcher : LinearSketch
}

impl ModelSpace {
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

    pub fn new(in_dimensions : usize, out_dimensions : usize) -> ModelSpace {
        let rc_feature_collections = get_rc_feature_collections(in_dimensions);
        let total_feat_dims = get_total_feat_dims(&rc_feature_collections);

        info!("And feature dims {}", total_feat_dims);

        let embedding_dim = total_feat_dims * out_dimensions;
        let sketched_embedding_dim = get_reduced_output_dimension(embedding_dim);
        let alpha = sketch_alpha(embedding_dim);

        let output_sketch = LinearSketch::new(embedding_dim, sketched_embedding_dim, alpha);

        let model_space = ModelSpace {
            in_dimensions : in_dimensions,
            feature_dimensions : total_feat_dims,
            out_dimensions : out_dimensions,
            feature_collections : rc_feature_collections,
            models : HashMap::new(),
            func_sketcher : output_sketch
        };
        model_space
    }

    //Samples a function applied to a bare vec
    pub fn thompson_sample_vec(&self, rng : &mut ThreadRng, other : &VectorSpace, target : &InverseSchmear) ->
                              (ModelKey, Array1<f32>, f32) {
        let mut result_key : ModelKey = 0 as ModelKey;
        let mut result_vec : Array1<f32> = Array::zeros((self.in_dimensions,));
        let mut result_dist = f32::INFINITY;
        for (key, model) in self.models.iter() {
            //Sample an array from the model
            let sample : SampledFunction = model.sample(rng);
            let (temp, temp_dist) = other.get_best_vector_arg(&sample, target);
            if (temp_dist < result_dist) {
                result_vec = temp;
                result_dist = temp_dist;
                result_key = *key;
            }
        }
        (result_key, result_vec, result_dist)
    }

    //Samples a bare term
    pub fn thompson_sample_term(&self, rng : &mut ThreadRng, inv_schmear : &InverseSchmear) -> (ModelKey, f32) {
        let mut result_key : ModelKey = 0 as ModelKey;
        let mut result_dist = f32::INFINITY;
        for (key, model) in self.models.iter() {
            //Sample a vector from the model
            let sample : Array1<f32> = model.sample_as_vec(rng);
            let compressed_sample = self.func_sketcher.sketch(&sample);
            let model_dist = inv_schmear.mahalanobis_dist(&compressed_sample);
            if (model_dist <= result_dist) {
                result_key = *key;
                result_dist = model_dist;
            }
        }
        (result_key, result_dist)
    }

    //Samples an application of terms
    pub fn thompson_sample_app(&self, rng : &mut ThreadRng, other : &ModelSpace, inv_schmear : &InverseSchmear) ->
                              (ModelKey, ModelKey, f32) {
        let mut result_func_key : ModelKey = 0 as ModelKey;
        let mut result_arg_key : ModelKey = 0 as ModelKey;
        let mut result_dist = f32::INFINITY;
        for (func_key, func_model) in self.models.iter() {
            //Sample a function from the function distribution
            let func_sample : SampledFunction = func_model.sample(rng);
            for (arg_key, arg_model) in other.models.iter() {
                let raw_arg_sample : Array1<f32> = arg_model.sample_as_vec(rng);
                let arg_sample : Array1<f32> = other.func_sketcher.sketch(&raw_arg_sample);

                let result : Array1<f32> = func_sample.apply(&arg_sample);
                let model_dist = inv_schmear.mahalanobis_dist(&result);
                if (model_dist < result_dist) {
                    result_func_key = *func_key;
                    result_arg_key = *arg_key;
                    result_dist = model_dist;
                }
            }
        }
        (result_func_key, result_arg_key, result_dist)
    }

    pub fn schmear_to_prior(&self, embedder_state : &EmbedderState, 
                        func_ptr : &TermPointer, in_schmear : &Schmear) -> NormalInverseWishart {
        let s : usize = self.feature_dimensions;
        let t : usize = self.out_dimensions;

        let mean_flat = self.func_sketcher.expand(&in_schmear.mean);
        let mean = mean_flat.into_shape((t, s)).unwrap();

        let big_sigma = FuncScatterTensor::from_compressed_covariance(t, s, 
                                                                  &self.func_sketcher, &in_schmear.covariance);
        let little_sigma = big_sigma.scale;

        let in_sigma = &big_sigma.in_scatter;
        let in_sigma_inv = pseudoinverse_h(in_sigma);

        let out_sigma = big_sigma.out_scatter;

        let dest_model = embedder_state.get_embedding(func_ptr);
        let single_observation_weight = dest_model.get_single_observation_weight();

        let in_weight = single_observation_weight * (1.0f32 / little_sigma).sqrt();
        let out_weight = single_observation_weight * little_sigma.sqrt();

        
        let mut in_precision = in_sigma_inv;
        in_precision *= in_weight;

        let mut big_v = out_sigma;
        big_v *= out_weight;

        let little_v = (t as f32) + 1.0f32;

        NormalInverseWishart::new(mean, in_precision, big_v, little_v)
    }

    fn get_feature_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    pub fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
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

    pub fn add_model(&mut self, model_key : ModelKey) {
        let model = Model::new(Rc::clone(&self.feature_collections), self.in_dimensions, self.out_dimensions);
        self.models.insert(model_key, model);
    }
    
    pub fn get_model_mut(&mut self, model_key : ModelKey) -> &mut Model {
        self.models.get_mut(&model_key).unwrap()
    }
    pub fn get_model(&self, model_key : ModelKey) -> &Model {
        self.models.get(&model_key).unwrap()
    }
    pub fn has_model(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
}
