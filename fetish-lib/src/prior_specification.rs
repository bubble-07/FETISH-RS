pub trait PriorSpecification {
    ///Returns the scaling factor to apply to the prior input precision
    fn get_in_precision_multiplier(&self, feat_dims : usize) -> f32;

    ///Returns the scaling factor to apply to the prior output covariance
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32;

    ///Returns the value for little_v, the number of observations of output covariance
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32;
}
