extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use rand::prelude::*;
use rand_distr::{Cauchy, Distribution, Gamma};
use rand_distr::StandardNormal;
use crate::params::*;

fn generate_standard_normal_random<R : Rng + ?Sized>(rng : &mut R, dims : usize) -> Array1<f32> {
    let as_vec : Vec<f32> = rng.sample_iter(StandardNormal).take(dims).collect();

    Array::from(as_vec)
}

fn generate_nsphere_random<R : Rng + ?Sized>(rng : &mut R, dims : usize) -> Array1<f32> {
    let mut vec = generate_standard_normal_random(rng, dims);
    let mut norm = 0.0;
    for i in 0..dims {
        norm += vec[[i,]] * vec[[i,]]
    }
    let norm = norm.sqrt();

    vec /= norm;

    vec
}

///Generates a zero-centered Cauchy random vector with the given `scaling` parameter and
///the given number of dimensions.
pub fn generate_cauchy_random<R : Rng + ?Sized>(rng : &mut R, scaling : f32, dims : usize) -> Array1<f32> {
    let cauchy = Cauchy::<f32>::new(0.0, scaling).unwrap();
    let norm : f32 = cauchy.sample(rng);
    let mut result = generate_nsphere_random(rng, dims);
    result *= norm;

    result
}

fn generate_inverse_gamma_random<R : Rng + ?Sized>(rng : &mut R, a : f32, b : f32) -> f32 {
    let gamma = Gamma::<f32>::new(a, b).unwrap();
    let inv_result = gamma.sample(rng);
    (1.0 / inv_result)
}

///Generates an inverse-gamma random variable with the given `a` and `b` parameters.
pub fn gen_inverse_gamma_random(rng : &mut ThreadRng, a : f32, b : f32) -> f32 {
    generate_inverse_gamma_random(rng, a, b)
}

///Generates a random vector on the unit `dims`-sphere.
pub fn gen_nsphere_random(rng : &mut ThreadRng, dims : usize) -> Array1<f32> {
    generate_nsphere_random(rng, dims)
}

///Generates a random vector contained in the unit `dims`-ball.
pub fn gen_nball_random(rng : &mut ThreadRng, dims : usize) -> Array1<f32> {
    let mut result = gen_nsphere_random(rng, dims);
    let u : f32 = rng.gen();
    let exponent = 1.0f32 / (dims as f32);
    let r = u.powf(exponent);
    result *= r;
    result
}
