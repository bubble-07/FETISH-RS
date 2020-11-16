use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;
use std::collections::HashMap;
use std::rc::*;
use crate::space_info::*;
use crate::term_pointer::*;
use crate::type_id::*;
use crate::linear_expression::*;
use crate::sampled_term_embedding::*;
use crate::term_reference::*;
use crate::array_utils::*;
use crate::holed_application::*;

pub struct SampledEmbedderState {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace>
}

impl SampledEmbedderState {
    pub fn has_embedding(&self, term_ptr : &TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    pub fn get_raw_embedding(&self, term_ptr : &TermPointer) -> &Array2<f32> {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_raw_embedding(term_ptr.index)
    }
    pub fn get_space_info(&self, type_id : &TypeId) -> Rc<SpaceInfo> {
        let result = &self.embedding_spaces.get(type_id).unwrap().space_info;
        Rc::clone(result)
    }

    fn inflate_embedding(&self, type_id : TypeId, compressed_embedding : Array1<f32>) -> SampledTermEmbedding {
        if (is_vector_type(type_id)) {
            SampledTermEmbedding::VectorEmbedding(compressed_embedding)
        } else {
            let space_info = self.get_space_info(&type_id);
            let inflated = space_info.inflate_compressed_vector(&compressed_embedding);
            SampledTermEmbedding::FunctionEmbedding(space_info, inflated)
        }
    }

    pub fn get_embedding(&self, term_ref : &TermReference) -> SampledTermEmbedding {
        match (term_ref) {
            TermReference::VecRef(vec) => SampledTermEmbedding::VectorEmbedding(from_noisy(vec)),
            TermReference::FuncRef(term_ptr) => {
                let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
                space.get_embedding(term_ptr.index)
            }
        }
    }

    pub fn evaluate_linear_expression(&self, linear_expr : &LinearExpression) -> SampledTermEmbedding {
        let mut current_embedding = self.get_embedding(&linear_expr.cap);
        //Now process through all of the holed applications in order
        for holed_application in linear_expr.chain.chain.iter().rev() {
            let ret_type = holed_application.get_type();
            let compressed_result = match (holed_application) {
                HoledApplication::FunctionHoled(arg_ref, ret_type) => {
                    match (current_embedding) {
                        SampledTermEmbedding::VectorEmbedding(vec) => { panic!(); },
                        SampledTermEmbedding::FunctionEmbedding(space_info, func_mat) => {
                            let arg_embedding = self.get_embedding(arg_ref);
                            let compressed_arg = arg_embedding.get_compressed();
                            let result = space_info.apply(&func_mat, &compressed_arg);
                            result
                        }
                    }
                },
                HoledApplication::ArgumentHoled(func_ptr) => {
                    let raw_func_embedding = self.get_raw_embedding(func_ptr);
                    let space_info = self.get_space_info(&func_ptr.type_id);
                    let compressed_arg = current_embedding.get_compressed();
                    let result = space_info.apply(raw_func_embedding, &compressed_arg);
                    result
                }
            };
            current_embedding = self.inflate_embedding(ret_type, compressed_result);
        }
        current_embedding
    }
}
