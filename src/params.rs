pub const CAUCHY_SCALING : f32 = 1.0;

pub const FOURIER_COVERAGE_MULTIPLIER : usize = 3;
pub const FOURIER_IMPORTANCE : f32 = 0.1f32;

pub const QUAD_PADDING_MULTIPLIER : usize = 1;
pub const QUAD_IMPORTANCE : f32 = 0.1f32;

pub const LIN_IMPORTANCE : f32 = 0.8f32;

pub const DIM : usize = 2;

pub const DIM_TAPER_START : usize = 4;

pub const TRAINING_POINTS_PER_ITER : usize = 5;

pub const OPT_MAX_ITERS : usize = 10;

pub const TARGET_INV_SCHMEAR_SCALE_FAC : f32 = 0.001f32;

//Priors for function optimum state
//Should be relatively big to reflect our strong belief in model misspecification
//in this case, due to the fact that optima may not be continuous w.r.t. function params
pub const FUNC_OPTIMUM_ERROR_COVARIANCE_PRIOR_OBSERVATIONS_PER_DIMENSION : f32 = 10.0f32;
//Should be pretty big
pub const FUNC_OPTIMUM_OUT_COVARIANCE_MULTIPLIER : f32 = 10.0f32;
//Should be pretty small, to reflect how little we know about the trend
pub const FUNC_OPTIMUM_IN_PRECISION_MULTIPLIER : f32 = 0.01f32;

//Priors for elaborator
//
pub const ELABORATOR_ERROR_COVARIANCE_PRIOR_OBSERVATIONS_PER_DIMENSION : f32 = 1.0f32;
//Should be pretty big (since we don't believe the subspace will necessarily be consistent)
pub const ELABORATOR_OUT_COVARIANCE_MULTIPLIER : f32 = 10.0f32;
//Should be pretty small (since we have no idea what the map should look like)
pub const ELABORATOR_IN_PRECISION_MULTIPLIER : f32 = 0.01f32;

//Priors for term models
//
//Should be pretty small (since we believe that there will be little model misspecification)
pub const TERM_MODEL_OUT_COVARIANCE_MULTIPLIER : f32 = 0.01f32;
//The larger this is, the more regularization in models. Should be moderately-sized.
pub const TERM_MODEL_IN_PRECISION_MULTIPLIER : f32 = 10.0f32;

//Optimization algorithm constants
pub const GAMMA : f32 = 0.95f32;
pub const LAMBDA : f32 = 1.0f32;

pub const RANDOM_VECTORS_PER_ITER : usize = 5;

pub const INITIAL_VALUE_FIELD_VARIANCE : f32 = 0.01f32;

pub const NUM_CONSTRAINT_REPEATS : usize = 3;

//Numerical algorithm constants
pub const NUM_STEEPEST_DESCENT_STEPS_PER_ITER : usize = 10;

pub const PINV_TRUNCATION_THRESH : f32 = 0.0001f32;

pub const UPDATE_SQ_NORM_TRUNCATION_THRESH : f32 = 0.0000001f32;

pub const DEFAULT_TEST_THRESH : f32 = 0.001f32;

pub fn log_tapered_linear(k : usize, x : usize) -> usize {
    if (x < k) {
        x
    } else {
        let k_float = k as f32;
        let x_float = x as f32;
        let result_float = (x_float / k_float).ln() * k_float + k_float;
        let ret = result_float.ceil() as usize;
        ret
    }
}

pub fn get_reduced_dimension(full_dim : usize) -> usize {
    log_tapered_linear(DIM_TAPER_START, full_dim)
}

pub fn num_fourier_features(in_dimension : usize) -> usize {
    FOURIER_COVERAGE_MULTIPLIER * get_reduced_dimension(in_dimension)
}
 
pub fn num_quadratic_features(in_dimension : usize) -> usize {
    let demanded_dims : usize = QUAD_PADDING_MULTIPLIER * in_dimension * in_dimension;
    get_reduced_dimension(demanded_dims)
}

pub fn num_sketched_linear_features(in_dimension : usize) -> usize {
    get_reduced_dimension(in_dimension)
}
