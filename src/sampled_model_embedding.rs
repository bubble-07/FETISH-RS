use ndarray::*;
use rand::prelude::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::schmear::*;
use crate::term_model::*;
use crate::inverse_schmear::*;
use crate::array_utils::*;
use crate::model::*;
use crate::space_info::*;

///Contains a bunch of useful information which originates
///from drawing a sample from a [`TermModel`].
pub struct SampledModelEmbedding {
    ///The [`FuncSchmear`] for the [`TermModel`] that this was sampled from.
    pub func_schmear : FuncSchmear,
    ///The [`FuncInverseSchmear`] for the [`TermModel`] that this was sampled from.
    pub func_inv_schmear : FuncInverseSchmear,
    ///The compressed-space [`Schmear`] for the [`TermModel`] that this was sampled from.
    pub compressed_schmear : Schmear,
    ///The compressed-space [`InverseSchmear`] for the [`TermModel`] that this was sampled from.
    pub compressed_inv_schmear : InverseSchmear,
    ///A sample from the underlying MNIW distribution for the [`TermModel`] that this was sampled from.
    pub sampled_mat : Array2<f32>,
    ///A flattened version of `sampled_mat`.
    pub sampled_vec : Array1<f32>,
    ///A compressed version version of `sampled_vec`.
    pub sampled_compressed_vec : Array1<f32>,
    ///A featurized version version of `sampled_compressed_vec`.
    pub sampled_feat_vec : Array1<f32>
}

impl SampledModelEmbedding {
    ///Draws a [`SampledModelEmbedding`] for the given [`TermModel`].
    pub fn new(term_model : &TermModel, rng : &mut ThreadRng) -> SampledModelEmbedding {
        let model = &term_model.model;
        let sampled_mat = model.sample(rng);
        let func_schmear = model.get_schmear(); 
        let func_inv_schmear = model.get_inverse_schmear();
        let sampled_vec = flatten_matrix(sampled_mat.view()).to_owned();

        let func_feat_info = model.get_context().get_feature_space_info(term_model.get_type_id());
        let projection_mat = func_feat_info.get_projection_matrix();

        let compressed_schmear = func_schmear.compress(projection_mat.view());
        let compressed_inv_schmear = compressed_schmear.inverse();
        let sampled_compressed_vec = projection_mat.dot(&sampled_vec);

        let sampled_feat_vec = func_feat_info.get_features(sampled_compressed_vec.view());

        SampledModelEmbedding {
            func_schmear,
            func_inv_schmear,
            compressed_schmear,
            compressed_inv_schmear,
            sampled_mat,
            sampled_vec,
            sampled_compressed_vec,
            sampled_feat_vec
        }
    }
}
