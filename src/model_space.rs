extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::bayes_utils::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::sampled_function::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use arraymap::ArrayMap;
use rand::prelude::*;

use std::collections::HashMap;

type ModelKey = usize;

pub struct ModelSpace {
    in_dimensions : usize,
    feature_dimensions : usize,
    out_dimensions : usize,
    feature_collections : Rc<[EnumFeatureCollection; 3]>,
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

    pub fn compress_inverse_schmear(&self, inv_schmear : &InverseSchmear) -> InverseSchmear {
        self.func_sketcher.compress_inverse_schmear(inv_schmear)
    }

    pub fn new(in_dimensions : usize, out_dimensions : usize) -> ModelSpace {
        let feature_collections = get_feature_collections(in_dimensions);
        let rc_feature_collections = Rc::new(feature_collections);

        let mut total_feat_dims : usize = 0;
        for collection in rc_feature_collections.iter() {
            total_feat_dims += collection.get_dimension();
        }
        println!("And feature dims {}", total_feat_dims);

        let embedding_dim = total_feat_dims * out_dimensions;
        let sketched_embedding_dim = get_reduced_output_dimension(embedding_dim);
        let output_sketch = LinearSketch::new(embedding_dim, sketched_embedding_dim);

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
    pub fn thompson_sample_vec(&self, rng : &mut ThreadRng, inv_schmear : &InverseSchmear) -> 
                              (ModelKey, Array1<f32>, f32) {
        let mut result_key : ModelKey = 0 as ModelKey;
        let mut result_vec : Array1<f32> = Array::zeros((self.in_dimensions,));
        let mut result_dist = f32::INFINITY;
        for (key, model) in self.models.iter() {
            //Sample an array from the model
            let sample : SampledFunction = model.sample(rng);
            let (arg_val, dist) = sample.get_closest_arg_to_target(inv_schmear.clone());

            if (dist < result_dist) {
                result_key = *key;
                result_vec = arg_val;
                result_dist = dist;
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
            let model_dist = inv_schmear.mahalanobis_dist(&sample);
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
                let arg_sample : Array1<f32> = arg_model.sample_as_vec(rng);
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

    pub fn schmear_to_prior(&self, in_schmear : &Schmear) -> NormalInverseGamma {
        let expanded_schmear = self.func_sketcher.expand_schmear(in_schmear);
        let (mean, covar) = schmear_to_tensors(self.feature_dimensions, self.out_dimensions, &expanded_schmear);
        let sigma = FuncScatterTensor::from_four_tensor(&covar);
        let precision = sigma.inverse();

        let s : usize = self.feature_dimensions;
        let t : usize = self.out_dimensions;
        let a : f32 = ((t * (s - 1)) as f32) * -0.5f32;
        let b : f32 = 0.0f32;
        NormalInverseGamma::new(mean, precision, a, b, t, s)
    }

    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        to_jacobian(&self.feature_collections, in_vec)
    }

    fn get_features(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    pub fn apply_schmears(&self, f : &FuncSchmear, x : &Schmear) -> Schmear {
        self.compute_out_schmear(&f.mean, &f.covariance, x)
    }

    fn compute_out_schmear(&self, f_mean : &Array2<f32>, f_covar : &FuncScatterTensor,
                           x : &Schmear) -> Schmear {
        let x_mean = &x.mean;
        let x_covar = &x.covariance;

        let feat_vec = self.get_features(&x_mean);
        let jacobian = self.get_jacobian(&x_mean);

        //There are two terms here for covariance -- J_f(x) sigma_x J_f(x)^T
        let data_contrib = jacobian.dot(x_covar).dot(&jacobian.t());

        //and the double-contraction of sigma_f by featurized x's
        let feat_outer = outer(&feat_vec, &feat_vec);
        let model_contrib = f_covar.transform_in_out(&feat_outer);
       
        let out_covar = data_contrib + model_contrib;
        let out_mean = f_mean.dot(x_mean);

        Schmear {
            mean : out_mean,
            covariance : out_covar
        }
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
