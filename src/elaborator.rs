use fetish_lib::everything::*;
use crate::params::*;

pub struct ElaboratorPrior {
}
impl PriorSpecification for ElaboratorPrior {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        ELABORATOR_IN_PRECISION_MULTIPLIER
    }
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32 { 
        //We'll directly tinker with the mean covariance schmear's size
        let pseudo_observations = self.get_out_pseudo_observations(out_dims);
        pseudo_observations * ELABORATOR_OUT_COVARIANCE_MULTIPLIER
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        //The +4 is because we need to ensure that we always have
        //a valid covariance schmear for this model. See Wikipedia
        //page on the Inverse-Wishart distribution's variance
        (out_dims as f32) * ELABORATOR_ERROR_COVARIANCE_PRIOR_OBSERVATIONS_PER_DIMENSION + 4.0f32
    }
}

