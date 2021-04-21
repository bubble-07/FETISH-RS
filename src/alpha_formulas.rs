use fetish_lib::everything::*;
use crate::params::*;

//TODO: Verify that this is all valid

fn prior_sigma() -> f32 {
    (TERM_MODEL_OUT_COVARIANCE_MULTIPLIER / TERM_MODEL_IN_PRECISION_MULTIPLIER).sqrt()
}

pub fn sketch_alpha(embedding_dim : usize) -> f32 {
    (1.0f32 / prior_sigma()) * (2.0f32 / (embedding_dim as f32)).sqrt()
}

pub fn linear_sketched_alpha(full_dim : usize, sketch_dim : usize) -> f32 {
    LIN_IMPORTANCE * (2.0f32 / prior_sigma()) * (1.0f32 / ((full_dim * sketch_dim) as f32).sqrt())
}

pub fn fourier_sketched_alpha(fourier_feats : usize) -> f32 {
    FOURIER_IMPORTANCE * (1.0f32 / (fourier_feats as f32).sqrt())
}

pub fn quadratic_sketched_alpha(full_dim : usize) -> f32 {
    QUAD_IMPORTANCE * (2.0f32 / ((full_dim as f32) * prior_sigma() * prior_sigma()))
}
