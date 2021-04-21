use fetish_lib::everything::*;
use ndarray::*;
use ndarray_linalg::*;
use crate::sampled_value_field_state::*;

pub fn get_best_term_index_to_pass_with_value(embedding_space : &SampledEmbeddingSpace, 
                                        func_mat : ArrayView2<f32>, ret_type : TypeId,
                                        value_field_state : &SampledValueFieldState)
                                     -> Option<(TermIndex, TypedVector, f32)> {
    let mut best_arg_index = Option::None;
    let mut best_ret_vec = Option::None;
    let mut best_arg_value = f32::NEG_INFINITY;

    for (arg_index, arg_model) in embedding_space.models.iter() {
        let arg_vec = &arg_model.sampled_feat_vec; 
        let compressed_ret_vec = func_mat.dot(arg_vec);
        let typed_ret_vec = TypedVector {
            vec : compressed_ret_vec,
            type_id : ret_type
        };
        let value = value_field_state.get_value_for_compressed_vector(&typed_ret_vec);
        if (value > best_arg_value || best_ret_vec.is_none()) {
            best_arg_value = value;
            best_ret_vec = Option::Some(typed_ret_vec);
            best_arg_index = Option::Some(*arg_index);
        }
    }
    if (best_arg_index.is_none()) { 
        Option::None
    } else {
        Option::Some((best_arg_index.unwrap(), best_ret_vec.unwrap(), best_arg_value))
    }
}

pub fn get_best_term_index_to_apply_with_value(embedding_space : &SampledEmbeddingSpace, 
                                     featurized_arg_vector : ArrayView1<f32>, 
                                     ret_type : TypeId, value_field_state : &SampledValueFieldState) 
                                     -> Option<(TermIndex, TypedVector, f32)> {
    let mut best_model_index = Option::None;
    let mut best_compressed_vec = Option::None;
    let mut best_model_value = f32::NEG_INFINITY;

    for (model_index, model) in embedding_space.models.iter() {
        let mat = &model.sampled_mat;
        let compressed_ret_vec = mat.dot(&featurized_arg_vector);
        let typed_ret_vec = TypedVector {
            vec : compressed_ret_vec,
            type_id : ret_type
        };
        let value = value_field_state.get_value_for_compressed_vector(&typed_ret_vec);
        if (value > best_model_value || best_compressed_vec.is_none()) {
            best_model_value = value;
            best_compressed_vec = Option::Some(typed_ret_vec);
            best_model_index = Option::Some(*model_index);
        }
    }
    if (best_model_index.is_none()) {
        Option::None
    } else {
        Option::Some((best_model_index.unwrap(), best_compressed_vec.unwrap(), best_model_value))
    }
}
