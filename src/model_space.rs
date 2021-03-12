extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use rand::prelude::*;
use crate::sampled_model_embedding::*;
use crate::sigma_points::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::alpha_formulas::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::rand_utils::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::term_model::*;
use crate::space_info::*;
use crate::params::*;
use crate::type_id::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use rand::prelude::*;
use crate::function_space_info::*;
use crate::sampled_embedding_space::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

type ModelKey = usize;

pub struct ModelSpace {
    pub type_id : TypeId,
    pub models : HashMap<ModelKey, TermModel>
}

impl ModelSpace {
    pub fn get_random_model_key(&self, rng : &mut ThreadRng) -> ModelKey {
        let num_entries = self.models.len();
        let rand_usize : usize = rng.gen();
        let entry_index = rand_usize % num_entries;
        let mut i = 0;
        for model_key in self.models.keys() {
            if (i == entry_index) {
                return *model_key;
            }
            i += 1;
        }
        panic!();
    }

    pub fn sample(&self, rng : &mut ThreadRng) -> SampledEmbeddingSpace {
        let mut result = SampledEmbeddingSpace::new(self.type_id);
        for (key, model) in self.models.iter() {
            let sample = SampledModelEmbedding::new(&model.model, rng);
            result.models.insert(*key, sample);
        }
        result
    }

    pub fn schmear_to_prior(&self, embedder_state : &EmbedderState, 
                        func_ptr : &TermPointer, in_schmear : &Schmear) -> NormalInverseWishart {
        let func_space_info = get_function_space_info(self.type_id);
        let func_feat_info = get_feature_space_info(self.type_id);
        let s = func_space_info.get_feature_dimensions();
        let t = func_space_info.get_output_dimensions();

        let mean = inflate_compressed_vector(self.type_id, &in_schmear.mean);

        //The (t * s) x (t * s) big sigma
        let big_sigma = func_feat_info.expand_covariance(&in_schmear.covariance);
        //t x s x t x s
        let big_sigma_tensor = big_sigma.into_shape((t, s, t, s)).unwrap();
        let big_sigma_tensor_reordered = big_sigma_tensor.permuted_axes([0, 2, 1, 3]);
        let big_sigma_tensor_standard = big_sigma_tensor_reordered.as_standard_layout();
        //(t * t) x (s * s)
        let big_sigma_matrix = big_sigma_tensor_standard.into_shape((t * t, s * s)).unwrap();

        let dest_model = embedder_state.get_embedding(func_ptr);

        let existing_big_v = &dest_model.model.data.big_v;
        let mut existing_big_v_inv = pseudoinverse_h(existing_big_v);
        existing_big_v_inv *= (dest_model.model.data.little_v - (t as f32) - 1.0f32) / (t as f32); 

        let existing_big_v_inv_flat = existing_big_v_inv.into_shape((t * t,)).unwrap();

        //(s * s)
        let result_in_covariance_flat = existing_big_v_inv_flat.dot(&big_sigma_matrix);
        let result_in_covariance = result_in_covariance_flat.into_shape((s, s)).unwrap();
        let in_precision = pseudoinverse_h(&result_in_covariance);

        let big_v = Array::zeros((t, t));
        let little_v = (t as f32) + 1.0f32;

        NormalInverseWishart::new(mean, in_precision, big_v, little_v)
    }
    pub fn add_model(&mut self, model_key : ModelKey) {
        let model = TermModel::new(self.type_id);
        self.models.insert(model_key, model);
    }
    
    pub fn get_model_mut(&mut self, model_key : ModelKey) -> &mut TermModel {
        self.models.get_mut(&model_key).unwrap()
    }
    pub fn get_model(&self, model_key : ModelKey) -> &TermModel {
        self.models.get(&model_key).unwrap()
    }
    pub fn has_model(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }

    pub fn new(type_id : TypeId) -> ModelSpace {
        ModelSpace {
            models : HashMap::new(),
            type_id
        }
    }
}
