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
use crate::typed_vector::*;
use crate::sampled_value_field_state::*;
use crate::displayable_with_state::*;
use crate::context::*;

pub struct SampledEmbedderState<'a> {
    pub embedding_spaces : HashMap::<TypeId, SampledEmbeddingSpace<'a>>,
    pub ctxt : &'a Context
}

impl<'a> SampledEmbedderState<'a> {
    pub fn has_embedding(&self, term_ptr : TermPointer) -> bool {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.has_embedding(term_ptr.index)
    }
    pub fn get_model_embedding(&self, term_ptr : TermPointer) -> &SampledModelEmbedding {
        let space = self.embedding_spaces.get(&term_ptr.type_id).unwrap();
        space.get_embedding(term_ptr.index)
    }

    pub fn expand_compressed_function(&self, compressed_vec : &TypedVector) -> Array2<f32> {
        let space = self.embedding_spaces.get(&compressed_vec.type_id).unwrap();
        let result = space.expand_compressed_function(compressed_vec.vec.view());
        result
    }

    pub fn get_best_nonvector_application_with_value(&self, value_field_state : &SampledValueFieldState) 
                                                  -> (TermApplication, f32) {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_application = Option::None;
        
        for func_type_id in self.embedding_spaces.keys() {
            let arg_type_id = self.ctxt.get_arg_type_id(*func_type_id);
            let ret_type_id = self.ctxt.get_ret_type_id(*func_type_id);
            if (!self.ctxt.is_vector_type(arg_type_id) && !self.ctxt.is_vector_type(ret_type_id)) {
                let func_embedding_space = self.embedding_spaces.get(func_type_id).unwrap();

                for func_index in func_embedding_space.models.keys() {
                    let func_ptr = TermPointer {
                        type_id : *func_type_id,
                        index : *func_index
                    };

                    let arg_embedding_space = self.embedding_spaces.get(&arg_type_id).unwrap();

                    for arg_index in arg_embedding_space.models.keys() {
                        let arg_ptr = TermPointer {
                            type_id : arg_type_id,
                            index : *arg_index
                        };
                        let arg_ref = TermReference::FuncRef(arg_ptr);
                        let term_app = TermApplication {
                            func_ptr : func_ptr.clone(),
                            arg_ref
                        };

                        let ret_typed_vec = self.evaluate_term_application(&term_app);
                        
                        let value = value_field_state.get_value_for_compressed_vector(&ret_typed_vec);
                        if (value > best_value) {
                            best_value = value;
                            best_application = Option::Some(term_app);
                        }
                    }
                }
            }
        }
        (best_application.unwrap(), best_value)
    }

    pub fn get_best_term_to_apply(&self, compressed_arg_vector : &TypedVector,
                                         func_type_id : TypeId, value_field_state : &SampledValueFieldState)
                                  -> (TermPointer, TypedVector, f32) {
        let ret_type_id = self.ctxt.get_ret_type_id(func_type_id);
        let arg_feat_space = self.ctxt.get_feature_space_info(compressed_arg_vector.type_id);
        let featurized_arg_vector = arg_feat_space.get_features(compressed_arg_vector.vec.view());
        let func_embedding_space = self.embedding_spaces.get(&func_type_id).unwrap();
        let (func_index, ret_vec, value) = func_embedding_space.get_best_term_index_to_apply_with_value(
                                                featurized_arg_vector.view(),
                                                ret_type_id, value_field_state);
        let func_ptr = TermPointer {
            index : func_index,
            type_id : func_type_id
        };
        (func_ptr, ret_vec, value)
    }

    pub fn get_best_term_to_pass(&self, compressed_func_vector : &TypedVector, 
                                        value_field_state : &SampledValueFieldState)
                                -> (TermPointer, TypedVector, f32) {
        let func_mat = self.expand_compressed_function(compressed_func_vector);
        let arg_type = self.ctxt.get_arg_type_id(compressed_func_vector.type_id);
        let ret_type = self.ctxt.get_ret_type_id(compressed_func_vector.type_id);
        let arg_embedding_space = self.embedding_spaces.get(&arg_type).unwrap();
        let (arg_index, ret_compressed_vec, value) = arg_embedding_space.get_best_term_index_to_pass_with_value(
                                                 func_mat.view(), ret_type, value_field_state);
        let arg_ptr = TermPointer {
            index : arg_index,
            type_id : arg_type
        };

        (arg_ptr, ret_compressed_vec, value)
    }

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
