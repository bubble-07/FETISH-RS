extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::context::*;
use rand::prelude::*;
use crate::func_schmear::*;
use crate::prior_specification::*;
use crate::sampled_model_embedding::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::term_model::*;
use crate::space_info::*;
use crate::type_id::*;
use crate::schmear::*;
use crate::elaborator::*;
use crate::sampled_embedding_space::*;
use crate::term_index::*;
use serde::{Serialize, Deserialize};

use std::collections::HashMap;

///Collection of all learned information about embeddings for a given type
///in and [`EmbedderState`]. This comprises the learned [`Elaborator`]s and
///the learned [`TermModel`]s for the type.
pub struct EmbeddingSpace<'a> {
    pub type_id : TypeId,
    pub elaborator : Elaborator<'a>,
    pub models : HashMap<TermIndex, TermModel<'a>>,
    pub ctxt : &'a Context
}

#[derive(Serialize, Deserialize)]
pub struct SerializedEmbeddingSpace {
    pub type_id : TypeId,
    pub elaborator : SerializedElaborator,
    pub models : HashMap<TermIndex, SerializedTermModel>
}

impl SerializedEmbeddingSpace {
    pub fn deserialize<'a>(mut self, ctxt : &'a Context) -> EmbeddingSpace<'a> {
        let elaborator = self.elaborator.deserialize(ctxt);
        let mut models = HashMap::new();
        for (term_index, serialized_term_model) in self.models.drain() {
            let term_model = serialized_term_model.deserialize(ctxt);
            models.insert(term_index, term_model);
        }
        EmbeddingSpace {
            type_id : self.type_id,
            elaborator,
            models,
            ctxt
        }
    }
}

impl <'a> EmbeddingSpace<'a> {
    pub fn serialize(mut self) -> SerializedEmbeddingSpace {
        let elaborator = self.elaborator.serialize();
        let mut models = HashMap::new();
        for (term_index, deserialized_term_model) in self.models.drain() {
            let term_model = deserialized_term_model.serialize();
            models.insert(term_index, term_model);
        }
        SerializedEmbeddingSpace {
            type_id : self.type_id,
            elaborator,
            models
        }
    }

    ///Draws a sample from the distribution defined by this [`EmbeddingSpace`]
    ///over collections of [`TermModel`]s of the same type, to yield
    ///a corresponding `SampledEmbeddingSpace` containing information about
    ///sampled embeddings.
    pub fn sample(&self, rng : &mut ThreadRng) -> SampledEmbeddingSpace<'a> {
        //We do this for speed, but also because the variation should
        //already mostly be captured in the values for the embeddings
        //of various models. In light of that, this is taken as the MLE
        let elaborator = self.elaborator.get_mean();

        let mut result = SampledEmbeddingSpace::new(self.type_id, elaborator, self.ctxt);
        for (key, model) in self.models.iter() {
            let sample = SampledModelEmbedding::new(&model, rng);
            result.models.insert(*key, sample);
        }
        result
    }

    pub fn schmear_to_prior(&self, embedder_state : &EmbedderState, elaborator_func_schmear : &FuncSchmear,
                        func_ptr : TermPointer, in_schmear : &Schmear) -> NormalInverseWishart {
        let func_space_info = self.ctxt.get_function_space_info(self.type_id);
        let s = func_space_info.get_feature_dimensions();
        let t = func_space_info.get_output_dimensions();

        let full_flat_schmear = elaborator_func_schmear.apply(in_schmear);
        let mean = full_flat_schmear.mean.into_shape((t, s)).unwrap();

        //The (t * s) x (t * s) big sigma
        let big_sigma = full_flat_schmear.covariance;
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
    ///Adds a new [`TermModel`] with the assigned [`TermIndex`].
    pub fn add_model(&mut self, model_key : TermIndex) {
        let prior_spec = self.ctxt.get_model_prior_specification(self.type_id);
        let model = TermModel::new(self.type_id, prior_spec, self.ctxt);
        self.models.insert(model_key, model);
    }
    
    ///Gets a handle to the [`TermModel`] with the given [`TermIndex`].
    pub fn get_model_mut(&mut self, model_key : TermIndex) -> &mut TermModel<'a> {
        self.models.get_mut(&model_key).unwrap()
    }
    ///Gets a reference to the [`TermModel`] with the given [`TermIndex`].
    pub fn get_model(&self, model_key : TermIndex) -> &TermModel<'a> {
        self.models.get(&model_key).unwrap()
    }
    ///Determines if a [`TermModel`] exists with the given [`TermIndex`].
    pub fn has_model(&self, model_key : TermIndex) -> bool {
        self.models.contains_key(&model_key)
    }

    ///Constructs a new embedding space with the given [`TypeId`] for [`TermModel`]s
    ///and for the [`Elaborator`], respectively, and occurring within the given [`Context`].
    pub fn new(type_id : TypeId, ctxt : &'a Context) -> EmbeddingSpace<'a> {
        let elaborator = Elaborator::new(type_id, ctxt);
        EmbeddingSpace {
            models : HashMap::new(),
            elaborator,
            type_id,
            ctxt
        }
    }
}
