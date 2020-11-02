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
use crate::space_info::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use rand::prelude::*;
use crate::sampled_embedding_space::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

type ModelKey = usize;

pub struct ModelSpace {
    pub space_info : Rc<SpaceInfo>,
    models : HashMap<ModelKey, Model>
}

impl ModelSpace {
    pub fn new(in_dimensions : usize, out_dimensions : usize) -> ModelSpace {
        let space_info = SpaceInfo::new(in_dimensions, out_dimensions);
        ModelSpace::from_space_info(space_info)
    }

    fn from_space_info(space_info : SpaceInfo) -> ModelSpace {
        let model_space = ModelSpace {
            models : HashMap::new(),
            space_info : Rc::new(space_info)
        };
        model_space

    }

    pub fn sample(&self, rng : &mut ThreadRng) -> SampledEmbeddingSpace {
        let mut result = SampledEmbeddingSpace::new(Rc::clone(&self.space_info));
        for (key, model) in self.models.iter() {
            let sample = model.sample(rng);
            result.models.insert(*key, sample);
        }
        result
    }

    pub fn schmear_to_prior(&self, embedder_state : &EmbedderState, 
                        func_ptr : &TermPointer, in_schmear : &Schmear) -> NormalInverseWishart {
        let s : usize = self.space_info.feature_dimensions;
        let t : usize = self.space_info.out_dimensions;

        let mean_flat = self.space_info.func_sketcher.expand(&in_schmear.mean);
        let mean = mean_flat.into_shape((t, s)).unwrap();

        //The (t * s) x (t * s) big sigma
        let big_sigma = self.space_info.func_sketcher.expand_covariance(&in_schmear.covariance);
        //t x s x t x s
        let big_sigma_tensor = big_sigma.into_shape((t, s, t, s)).unwrap();
        let big_sigma_tensor_reordered = big_sigma_tensor.permuted_axes([0, 2, 1, 3]);
        let big_sigma_tensor_standard = big_sigma_tensor_reordered.as_standard_layout();
        //(t * t) x (s * s)
        let big_sigma_matrix = big_sigma_tensor_standard.into_shape((t * t, s * s)).unwrap();

        let dest_model = embedder_state.get_embedding(func_ptr);

        let existing_big_v = &dest_model.data.big_v;
        let mut existing_big_v_inv = pseudoinverse_h(existing_big_v);
        existing_big_v_inv *= (dest_model.data.little_v - (t as f32) - 1.0f32) / (t as f32); 

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
        let model = Model::new(Rc::clone(&self.space_info));
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
