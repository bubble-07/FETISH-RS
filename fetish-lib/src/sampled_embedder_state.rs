use ndarray::*;
use crate::sampled_embedding_space::*;
use std::collections::HashMap;
use crate::space_info::*;
use crate::term_pointer::*;
use crate::type_id::*;
use crate::term_reference::*;
use crate::array_utils::*;
use crate::interpreter_state::*;
use crate::sampled_model_embedding::*;
use crate::term_application::*;
use crate::displayable_with_state::*;
use crate::typed_vector::*;
use crate::context::*;

///A sampled possible state for embeddings drawn from an [`crate::embedder_state::EmbedderState`].
///This [`SampledEmbedderState`] has one [`SampledEmbeddingSpace`] for every
///function [`TypeId`] in the given [`Context`], which in turn contains
///a sampled [`crate::elaborator::Elaborator`] and [`SampledModelEmbedding`]s
pub struct SampledEmbedderState<'a> {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace<'a>>,
    pub ctxt : &'a Context
}

impl<'a> SampledEmbedderState<'a> {
    ///Determines whether this [`SampledEmbedderState`] has an embedding for the
    ///given [`TermPointer`].
    pub fn has_embedding(&self, term_ptr : TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    ///Gets the [`SampledModelEmbedding`] for the given [`TermPointer`].
    pub fn get_model_embedding(&self, term_ptr : TermPointer) -> &SampledModelEmbedding {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_embedding(term_ptr.index)
    }

    ///Given a compressed [`TypedVector`] for a function type, expands the
    ///compressed vector using this [`SampledEmbedderState`]'s corresponding
    ///elaborator sample and inflates it to yield a linear transformation
    ///from the feature space of the input space to the compressed space of the output.
    pub fn expand_compressed_function(&self, compressed_vec : &TypedVector) -> Array2<f32> {
        let space = self.embedding_spaces.get(&compressed_vec.type_id).unwrap();
        let result = space.expand_compressed_function(compressed_vec.vec.view());
        result
    }

    ///Given a [`TermApplication`], evaluates the result that the [`SampledModelEmbedding`]s
    ///in this [`SampledEmbedderState`] would yield for the expression. The result will be
    ///in the compressed space of the output type.
    pub fn evaluate_term_application(&self, term_application : &TermApplication) -> TypedVector {
        let func_type_id = term_application.func_ptr.type_id;
        let ret_type_id = self.ctxt.get_ret_type_id(func_type_id);
        
        let func_space_info = self.ctxt.get_function_space_info(func_type_id);
        let func_embedding_space = self.embedding_spaces.get(&func_type_id).unwrap();
        let func_mat = &func_embedding_space.get_embedding(term_application.func_ptr.index).sampled_mat;

        let arg_vec = match (&term_application.arg_ref) {
            TermReference::VecRef(_, vec) => from_noisy(vec.view()),
            TermReference::FuncRef(arg_ptr) => {
                let arg_embedding_space = self.embedding_spaces.get(&arg_ptr.type_id).unwrap();
                arg_embedding_space.get_embedding(arg_ptr.index).sampled_compressed_vec.clone()
            }
        };

        let ret_vec = func_space_info.apply(func_mat.view(), arg_vec.view());
        TypedVector {
            vec : ret_vec,
            type_id : ret_type_id
        }
    }
}
