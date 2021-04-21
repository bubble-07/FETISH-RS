use fetish_lib::everything::*;
use crate::params::*;

pub struct TermModelPriorSpecification {
}

impl PriorSpecification for TermModelPriorSpecification {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        TERM_MODEL_IN_PRECISION_MULTIPLIER
    }
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32 {
        let pseudo_observations = self.get_out_pseudo_observations(out_dims);
        pseudo_observations * TERM_MODEL_OUT_COVARIANCE_MULTIPLIER
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        //Minimal number of pseudo-observations to have a defined
        //mean and covariance with no observations on the model
        (out_dims as f32) + 4.0f32
    }
}
