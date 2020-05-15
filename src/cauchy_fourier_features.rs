extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use rand::prelude::*;
use rand_distr::{Cauchy, Distribution};
use rand_distr::StandardNormal;

const CAUCHY_SCALING : f32 = 10.0;

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

fn generate_cauchy_random<R : Rng + ?Sized>(rng : &mut R, dims : usize) -> Array1<f32> {
    let cauchy = Cauchy::<f32>::new(0.0, CAUCHY_SCALING).unwrap();
    let norm : f32 = cauchy.sample(rng);
    let mut result = generate_nsphere_random(rng, dims);
    result *= norm;

    result
}
