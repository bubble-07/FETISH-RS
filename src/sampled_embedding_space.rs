use ndarray::*;
use std::collections::HashMap;
use crate::sampled_model_embedding::*;
use crate::space_info::*;
use crate::type_id::*;
use crate::context::*;
use crate::term_index::*;

type ModelKey = TermIndex;

///All information in a [`crate::sampled_embedder_state::SampledEmbedderState`] pertaining to a particular
///function [`TypeId`]
pub struct SampledEmbeddingSpace<'a> {
    pub type_id : TypeId,
    ///A sample drawn from the [`crate::elaborator::Elaborator`] for this type in the original 
    ///[`crate::embedder_state::EmbedderState`].
    ///Maps from the compressed space for the type to the base space for the type.
    pub elaborator : Array2<f32>,
    pub models : HashMap<ModelKey, SampledModelEmbedding>,
    pub ctxt : &'a Context
}

impl<'a> SampledEmbeddingSpace<'a> {
    ///Determines whether an embedding exists for the given [`TermIndex`].
    pub fn has_embedding(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
    ///Gets the [`SampledModelEmbedding`] corresponding to the given  [`TermIndex`].
    pub fn get_embedding(&self, model_key : ModelKey) -> &SampledModelEmbedding {
        self.models.get(&model_key).unwrap()
    }

    ///Given a compressed vector, uses `self.elaborator` to expand it to a vector in the
    ///base space of `self.type_id`.
    pub fn expand_compressed_vector(&self, compressed_vec : ArrayView1<f32>) -> Array1<f32> {
        let elaborated_vec = self.elaborator.dot(&compressed_vec);
        elaborated_vec
    }

    ///Given a compressed vector for a function of `self.type_id`, 
    ///first performs [`Self::expand_compressed_vector`] and
    ///then inflates the result to yield a linear transformation
    ///from the feature space of the input type to the compressed space of the output type.
    pub fn expand_compressed_function(&self, compressed_vec : ArrayView1<f32>) -> Array2<f32> {
        let func_space_info = self.ctxt.get_function_space_info(self.type_id);
        let feat_dims = func_space_info.get_feature_dimensions();
        let out_dims = func_space_info.get_output_dimensions();

        let elaborated_vec = self.expand_compressed_vector(compressed_vec);
        let result = elaborated_vec.into_shape((out_dims, feat_dims)).unwrap(); 
        result
    }

    ///Creates a new, initially-empty [`SampledEmbeddingSpace`].
    pub fn new(type_id : TypeId, elaborator : Array2<f32>, ctxt : &'a Context) -> SampledEmbeddingSpace<'a> {
        SampledEmbeddingSpace {
            type_id,
            elaborator,
            models : HashMap::new(),
            ctxt
        }
    }
}
