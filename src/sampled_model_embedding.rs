use ndarray::*;
use ndarray_linalg::*;
use crate::function_space_info::*;
use std::rc::*;
use std::collections::HashMap;
use rand::prelude::*;
use crate::sampled_term_embedding::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::array_utils::*;
use crate::model::*;
use crate::space_info::*;

//TODO: Store the featurized version of what's sampled, so that you don't have
//to re-compute that multiple times
pub struct SampledModelEmbedding {
    pub func_schmear : FuncSchmear,
    pub func_inv_schmear : FuncInverseSchmear,
    pub compressed_schmear : Schmear,
    pub compressed_inv_schmear : InverseSchmear,
    pub sampled_mat : Array2<f32>,
    pub sampled_vec : Array1<f32>,
    pub sampled_compressed_vec : Array1<f32>
}

impl SampledModelEmbedding {
    pub fn new(model : &Model, rng : &mut ThreadRng) -> SampledModelEmbedding {
        let sampled_mat = model.sample(rng);
        let func_schmear = model.get_schmear(); 
        let func_inv_schmear = model.get_inverse_schmear();
        let sampled_vec = flatten_matrix(&sampled_mat);

        let func_feat_info = get_feature_space_info(model.get_type_id());
        let projection_mat = func_feat_info.get_projection_matrix();

        let compressed_schmear = func_schmear.compress(&projection_mat);
        let compressed_inv_schmear = compressed_schmear.inverse();
        let sampled_compressed_vec = projection_mat.dot(&sampled_vec);

        SampledModelEmbedding {
            func_schmear,
            func_inv_schmear,
            compressed_schmear,
            compressed_inv_schmear,
            sampled_mat,
            sampled_vec,
            sampled_compressed_vec
        }
    }
}
