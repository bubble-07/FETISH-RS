use ndarray::*;
use fetish_lib::everything::*;
use crate::sampled_value_field_state::*;
use crate::sampled_embedding_space::*;

pub fn get_best_nonvector_application_with_value(state : &SampledEmbedderState, value_field_state : &SampledValueFieldState) 
                                              -> Option<(TermApplication, f32)> {
    let mut best_value = f32::NEG_INFINITY;
    let mut best_application = Option::None;
    
    for func_type_id in state.embedding_spaces.keys() {
        let arg_type_id = state.ctxt.get_arg_type_id(*func_type_id);
        let ret_type_id = state.ctxt.get_ret_type_id(*func_type_id);
        if (!state.ctxt.is_vector_type(arg_type_id) && !state.ctxt.is_vector_type(ret_type_id)) {
            let func_embedding_space = state.embedding_spaces.get(func_type_id).unwrap();

            for func_index in func_embedding_space.models.keys() {
                let func_ptr = TermPointer {
                    type_id : *func_type_id,
                    index : *func_index
                };

                let arg_embedding_space = state.embedding_spaces.get(&arg_type_id).unwrap();

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

                    let ret_typed_vec = state.evaluate_term_application(&term_app);
                    
                    let value = value_field_state.get_value_for_compressed_vector(&ret_typed_vec);
                    if (value > best_value) {
                        best_value = value;
                        best_application = Option::Some(term_app);
                    }
                }
            }
        }
    }
    if (best_application.is_none()) {
        Option::None
    } else {
        Option::Some((best_application.unwrap(), best_value))
    }
}

pub fn get_best_term_to_apply(state : &SampledEmbedderState, compressed_arg_vector : &TypedVector,
                                     func_type_id : TypeId, value_field_state : &SampledValueFieldState)
                              -> Option<(TermPointer, TypedVector, f32)> {
    let ret_type_id = state.ctxt.get_ret_type_id(func_type_id);
    let arg_feat_space = state.ctxt.get_feature_space_info(compressed_arg_vector.type_id);
    let featurized_arg_vector = arg_feat_space.get_features(compressed_arg_vector.vec.view());
    let func_embedding_space = state.embedding_spaces.get(&func_type_id).unwrap();
    let maybe_term_index = get_best_term_index_to_apply_with_value(func_embedding_space,
                                            featurized_arg_vector.view(),
                                            ret_type_id, value_field_state);
    if (maybe_term_index.is_none()) {
        return Option::None
    }
    let (func_index, ret_vec, value) = maybe_term_index.unwrap();
    let func_ptr = TermPointer {
        index : func_index,
        type_id : func_type_id
    };
    Option::Some((func_ptr, ret_vec, value))
}

pub fn get_best_term_to_pass(state : &SampledEmbedderState, compressed_func_vector : &TypedVector, 
                                    value_field_state : &SampledValueFieldState)
                            -> Option<(TermPointer, TypedVector, f32)> {
    let func_mat = state.expand_compressed_function(compressed_func_vector);
    let arg_type = state.ctxt.get_arg_type_id(compressed_func_vector.type_id);
    let ret_type = state.ctxt.get_ret_type_id(compressed_func_vector.type_id);
    let arg_embedding_space = state.embedding_spaces.get(&arg_type).unwrap();
    let maybe_term_index = get_best_term_index_to_pass_with_value(arg_embedding_space, 
                                             func_mat.view(), ret_type, value_field_state);
    if (maybe_term_index.is_none()) {
        return Option::None
    }
    let (arg_index, ret_compressed_vec, value) = maybe_term_index.unwrap();
    let arg_ptr = TermPointer {
        index : arg_index,
        type_id : arg_type
    };

    Option::Some((arg_ptr, ret_compressed_vec, value))
}
