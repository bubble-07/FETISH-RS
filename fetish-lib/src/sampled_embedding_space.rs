use ndarray::*;
use std::collections::HashMap;
use crate::sampled_model_embedding::*;
use crate::space_info::*;
use crate::type_id::*;
use crate::context::*;
use crate::term_index::*;

type ModelKey = TermIndex;

pub struct SampledEmbeddingSpace<'a> {
    pub type_id : TypeId,
    pub elaborator : Array2<f32>,
    pub models : HashMap<ModelKey, SampledModelEmbedding>,
    pub ctxt : &'a Context
}

impl<'a> SampledEmbeddingSpace<'a> {
    pub fn has_embedding(&self, model_key : ModelKey) -> bool {
        self.models.contains_key(&model_key)
    }
    pub fn get_embedding(&self, model_key : ModelKey) -> &SampledModelEmbedding {
        self.models.get(&model_key).unwrap()
    }

    pub fn expand_compressed_vector(&self, compressed_vec : ArrayView1<f32>) -> Array1<f32> {
        let elaborated_vec = self.elaborator.dot(&compressed_vec);
        elaborated_vec
    }

    pub fn expand_compressed_function(&self, compressed_vec : ArrayView1<f32>) -> Array2<f32> {
        let func_space_info = self.ctxt.get_function_space_info(self.type_id);
        let feat_dims = func_space_info.get_feature_dimensions();
        let out_dims = func_space_info.get_output_dimensions();

        let elaborated_vec = self.expand_compressed_vector(compressed_vec);
        let result = elaborated_vec.into_shape((out_dims, feat_dims)).unwrap(); 
        result
    }

    pub fn new(type_id : TypeId, elaborator : Array2<f32>, ctxt : &'a Context) -> SampledEmbeddingSpace<'a> {
        SampledEmbeddingSpace {
            type_id,
            elaborator,
            models : HashMap::new(),
            ctxt
        }
    }
}
