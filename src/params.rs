pub const CAUCHY_SCALING : f32 = 1.0;

pub const FOURIER_COVERAGE_MULTIPLIER : usize = 3;
pub const FOURIER_IMPORTANCE : f32 = 0.1f32;

pub const QUAD_PADDING_MULTIPLIER : usize = 1;
pub const QUAD_IMPORTANCE : f32 = 0.1f32;

pub const LIN_IMPORTANCE : f32 = 0.8f32;

pub const PRIOR_SIGMA : f32 = 1.0f32;

pub const DIM : usize = 2;

pub const DIM_TAPER_START : usize = 4;

pub const TRAINING_POINTS_PER_ITER : usize = 5;

//Optimization algorithm constants
pub const GAMMA : f32 = 0.95f32;
pub const LAMBDA : f32 = 1.0f32;

pub const RANDOM_VECTORS_PER_ITER : usize = 5;

pub const INITIAL_VALUE_FIELD_VARIANCE : f32 = 0.01f32;

pub const INITIAL_FUNCTION_OPTIMUM_VARIANCE : f32 = 20.0f32;

pub const NUM_CONSTRAINT_REPEATS : usize = 3;

//closer to 0.0 -> less myopic, 1.0 -> more myopic 
//(exponential moving average, for statistics on optimal arguments to vector-valued funcs)
pub const LERP_FACTOR : f32 = 0.25f32;

//Numerical algorithm constants
pub const NUM_STEEPEST_DESCENT_STEPS_PER_ITER : usize = 10;

pub const PINV_TRUNCATION_THRESH : f32 = 0.0001f32;

pub const FUNC_SCATTER_TENSOR_ZEROING_THRESH : f32 = 1e-15f32;

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
