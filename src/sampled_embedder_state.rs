use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_embedding_space::*;
use std::collections::HashMap;
use std::rc::*;
use crate::function_space_info::*;
use crate::term_pointer::*;
use crate::type_id::*;
use crate::sampled_term_embedding::*;
use crate::term_reference::*;
use crate::array_utils::*;
use crate::interpreter_state::*;
use crate::sampled_model_embedding::*;
use crate::term_application::*;
use crate::typed_vector::*;
use crate::value_field_state::*;
use crate::value_field::*;

pub struct SampledEmbedderState {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace>
}

impl SampledEmbedderState {
    pub fn has_embedding(&self, term_ptr : &TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    pub fn get_model_embedding(&self, term_ptr : &TermPointer) -> &SampledModelEmbedding {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_embedding(term_ptr.index)
    }
    pub fn get_space_info(&self, type_id : &TypeId) -> &FunctionSpaceInfo {
        let result = &self.embedding_spaces.get(type_id).unwrap().func_space_info;
        result
    }

    pub fn get_best_nonvector_application_with_value(&self, interpreter_state : &InterpreterState,
                                                            value_field_state : &ValueFieldState) 
                                                  -> (TermApplication, f32) {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_application = Option::None;
        
        for func_type_id in self.embedding_spaces.keys() {
            let arg_type_id = get_arg_type_id(*func_type_id);
            let ret_type_id = get_ret_type_id(*func_type_id);
            if (!is_vector_type(arg_type_id) && !is_vector_type(ret_type_id)) {
                let func_embedding_space = self.embedding_spaces.get(func_type_id).unwrap();
                let arg_embedding_space = self.embedding_spaces.get(&arg_type_id).unwrap();

                let application_table = interpreter_state.application_tables.get(func_type_id).unwrap();

                for func_index in func_embedding_space.models.keys() {
                    let func_ptr = TermPointer {
                        type_id : *func_type_id,
                        index : *func_index
                    };
                    let func_mat = &func_embedding_space.get_embedding(*func_index).sampled_mat;
                    let func_space_info = &func_embedding_space.func_space_info;

                    for arg_index in func_embedding_space.models.keys() {
                        let arg_ptr = TermPointer {
                            type_id : arg_type_id,
                            index : *arg_index
                        };
                        let arg_ref = TermReference::FuncRef(arg_ptr);
                        let term_app = TermApplication {
                            func_ptr : func_ptr.clone(),
                            arg_ref
                        };


                        if (!application_table.has_computed(&term_app)) {
                            let arg_vec = &arg_embedding_space.get_embedding(*arg_index).sampled_compressed_vec;
                            let ret_vec = func_space_info.apply(func_mat, arg_vec);
                            let ret_typed_vec = TypedVector {
                                vec : ret_vec,
                                type_id : ret_type_id
                            };

                            let value = value_field_state.get_value_for_vector(&ret_typed_vec);
                            if (value > best_value) {
                                best_value = value;
                                best_application = Option::Some(term_app);
                            }
                        }
                    }
                }
            }
        }
        (best_application.unwrap(), best_value)
    }

    fn inflate_embedding(&self, type_id : TypeId, compressed_embedding : Array1<f32>) -> SampledTermEmbedding {
        if (is_vector_type(type_id)) {
            SampledTermEmbedding::VectorEmbedding(compressed_embedding)
        } else {
            let space_info = self.get_space_info(&type_id);
            let inflated = space_info.inflate_compressed_vector(&compressed_embedding);
            SampledTermEmbedding::FunctionEmbedding(space_info.clone(), inflated)
        }
    }

    pub fn get_term_embedding(&self, term_ref : &TermReference) -> SampledTermEmbedding {
        match (term_ref) {
            TermReference::VecRef(vec) => SampledTermEmbedding::VectorEmbedding(from_noisy(vec)),
            TermReference::FuncRef(term_ptr) => {
                let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
                space.get_term_embedding(term_ptr.index)
            }
        }
    }
}
