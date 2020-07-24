pub const CAUCHY_SCALING : f32 = 1.0;

pub const FOURIER_REG_STRENGTH : f32 = 1.0;
pub const FOURIER_COVERAGE_MULTIPLIER : usize = 5;

pub const LIN_REG_STRENGTH : f32 = 0.1;

pub const QUAD_REG_STRENGTH : f32 = 5.0;
pub const QUAD_PADDING_MULTIPLIER : usize = 1;

pub const DIM : usize = 2;

pub const IN_TAPER_START : usize = 5;
pub const OUT_TAPER_START : usize = 5;

//Numerical algorithm constants
pub const SVD_PRECISION : f32 = 0.00001f32;
pub const SVD_MAX_ITER : usize = 1000;

pub const INITIAL_INV_GAMMA_A : f32 = 0.5f32;
pub const INITIAL_INV_GAMMA_B : f32 = 1.0f32;

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
