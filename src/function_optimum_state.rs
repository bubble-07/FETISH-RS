use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashMap;
use crate::term_pointer::*;
use crate::type_id::*;
use crate::typed_vector::*;
use crate::term_reference::*;
use crate::function_optimum_space::*;
use crate::term_application::*;
use crate::value_field_state::*;
use crate::sampled_embedder_state::*;
use crate::space_info::*;

pub struct FunctionOptimumState {
    pub function_spaces : HashMap::<TypeId, FunctionOptimumSpace>
}

impl FunctionOptimumState {
    pub fn new() -> FunctionOptimumState {
        let mut function_spaces = HashMap::new();
        for func_type_id in 0..total_num_types() {
            if (!is_vector_type(func_type_id)) {
                let arg_type = get_arg_type_id(func_type_id);
                let ret_type = get_ret_type_id(func_type_id);
                if (is_vector_type(arg_type) && !is_vector_type(ret_type)) {
                    let function_optimum_space = FunctionOptimumSpace::new(func_type_id);
                    function_spaces.insert(func_type_id, function_optimum_space); 
                } 
            }
        }
        FunctionOptimumState {
            function_spaces
        }
    }
    pub fn get_best_vector_to_pass(&self, compressed_func_vector : &TypedVector, 
                                   value_field_state : &ValueFieldState,
                                   sampled_embedder_state : &SampledEmbedderState)
                                   -> (Array1<f32>, TypedVector, f32) {

        let func_type_id = compressed_func_vector.type_id;
        let arg_type_id = get_arg_type_id(func_type_id);
        let ret_type_id = get_ret_type_id(func_type_id);

        let arg_feat_space = get_feature_space_info(arg_type_id);

        let func_optimum_space = self.function_spaces.get(&func_type_id).unwrap();
        let best_arg_vector = func_optimum_space.estimate_optimal_vector_for_compressed_func(
                                                 &compressed_func_vector.vec);

        let best_feat_vector = arg_feat_space.get_features(&best_arg_vector);
        let func_mat = sampled_embedder_state.expand_compressed_function(compressed_func_vector);
        let ret_vec = func_mat.dot(&best_feat_vector);
        let ret_typed_vec = TypedVector {
            vec : ret_vec,
            type_id : ret_type_id
        };
        let value = value_field_state.get_value_for_vector(&ret_typed_vec);

        (best_arg_vector, ret_typed_vec, value)
    }
    pub fn update(&mut self, sampled_embedder_state : &SampledEmbedderState, 
                             value_field_state : &ValueFieldState) -> (TermApplication, f32) {

        let mut best_term_app = Option::None;
        let mut best_value = f32::NEG_INFINITY;

        for (func_type_id, function_space) in self.function_spaces.iter_mut() {
            let sampled_embeddings = sampled_embedder_state.embedding_spaces.get(func_type_id).unwrap();

            let (func_index, value) = function_space.update(sampled_embeddings, value_field_state);

            if (value > best_value) {
                let func_ptr = TermPointer {
                    index : func_index,
                    type_id : *func_type_id
                };
                let arg_vec = function_space.get_optimal_vector(func_index);
                let arg_ref = TermReference::from(arg_vec);
                let term_app = TermApplication {
                    func_ptr,
                    arg_ref
                };

                best_value = value;
                best_term_app = Option::Some(term_app);
            }
        }
        (best_term_app.unwrap(), best_value)
    }
}
