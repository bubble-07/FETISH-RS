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
use crate::sampled_model_embedding::*;
use crate::function_optimum_space::*;

pub struct FunctionOptimumState {
    pub function_spaces : HashMap::<TypeId, FunctionOptimumSpace>
}
