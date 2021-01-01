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

pub const HIGHER_ORDER_PENALTY : f32 = 1.2f32;

pub const NUM_FUNCTION_SAMPLES : usize = 10;
pub const NUM_ELLIPSOID_SAMPLES : usize = 10;

//Numerical algorithm constants
pub const PINV_TRUNCATION_THRESH : f32 = 0.0001f32;

pub const FUNC_SCATTER_TENSOR_ZEROING_THRESH : f32 = 1e-15f32;

pub const DEFAULT_TEST_THRESH : f32 = 0.001f32;

pub const ENCLOSING_ELLIPSOID_DIRECTION_MULTIPLIER : usize = 4;
pub const ENCLOSING_ELLIPSOID_INITIAL_SCALE : f32 = 0.000001f32;
pub const ENCLOSING_ELLIPSOID_MAXIMAL_SCALE : f32 = 100000.0f32;
pub const ENCLOSING_ELLIPSOID_GROWTH_FACTOR : f32 = 2.0f32;
pub const ENCLOSING_ELLIPSOID_BRENT_REL_ERROR : f32 = 0.0001f32;
pub const ENCLOSING_ELLIPSOID_BRENT_MAX_ITERS : u64 = 100;

pub const QUEUE_TERMS_PER_ITER : usize = 100;

pub const EXPLORATION_TERMS_PER_ITER : usize = 10;

pub const NUM_OPT_ITERS : u64 = 1000;
pub const LBFGS_HISTORY : usize = 6;
pub const MORE_THUENTE_A : f32 = 1e-4;
pub const MORE_THUENTE_B : f32 = 0.9;

pub const ENCLOSING_ELLIPSOID_TOLERANCE : f32 = 0.001;
pub const ENCLOSING_ELLIPSOID_MAX_ITERS : usize = 1000;

pub const HEURISTIC_ELLIPSE_CI : f32 = 0.95;
pub const HEURISTIC_ELLIPSE_MAX_DIMENSION : usize = 5000;
pub const HEURISTIC_ELLIPSE_BRENT_REL_ERROR : f64 = 0.0001;

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
