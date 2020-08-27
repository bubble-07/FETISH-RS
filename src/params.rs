pub const CAUCHY_SCALING : f32 = 1.0;

pub const FOURIER_COVERAGE_MULTIPLIER : usize = 3;
pub const FOURIER_IMPORTANCE : f32 = 0.1f32;

pub const QUAD_PADDING_MULTIPLIER : usize = 1;
pub const QUAD_IMPORTANCE : f32 = 0.1f32;

pub const LIN_IMPORTANCE : f32 = 0.8f32;

pub const PRIOR_SIGMA : f32 = 1.0f32;

pub const DIM : usize = 2;

pub const IN_TAPER_START : usize = 4;
pub const OUT_TAPER_START : usize = 4;

pub const TRAINING_POINTS_PER_ITER : usize = 5;

//A value of 1.0 for a ensures that with at least one data-point,
//the expectation of the gamma is always defined
//but with zero it is now.
pub const INITIAL_INV_GAMMA_A : f32 = 1.0f32;

//This needs to be strictly greater than zero for covariance and precision
//to both be well-defined
pub const INITIAL_INV_GAMMA_B : f32 = 1.0f32;

//Numerical algorithm constants
pub const ZEROING_THRESH : f32 = 0.0001f32;

pub const NUM_OPT_ITERS : u64 = 20;
pub const LBFGS_HISTORY : usize = 6;

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

pub fn get_reduced_input_dimension(x : usize) -> usize {
    log_tapered_linear(IN_TAPER_START, x)
}

pub fn get_reduced_output_dimension(x : usize) -> usize {
    log_tapered_linear(OUT_TAPER_START, x)
}

pub fn num_fourier_features(in_dimension : usize) -> usize {
    FOURIER_COVERAGE_MULTIPLIER * get_reduced_input_dimension(in_dimension)
}
 
pub fn num_quadratic_features(in_dimension : usize) -> usize {
    let demanded_dims : usize = QUAD_PADDING_MULTIPLIER * in_dimension * in_dimension;
    get_reduced_input_dimension(demanded_dims)
}

pub fn num_sketched_linear_features(in_dimension : usize) -> usize {
    get_reduced_input_dimension(in_dimension)
}
