#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]
#![allow(unused_variables)]

mod sampled_embedder_state;
mod sampled_embedding_space;
mod space_info;
mod holed_application;
mod holed_linear_expression;
mod linear_expression;
mod data_update;
mod sigma_points;
mod sherman_morrison;
mod data_point;
mod normal_inverse_wishart;
mod normal_inverse_wishart_sampler;
mod wishart;
mod least_squares;
mod alpha_formulas;
mod test_utils;
mod array_utils;
mod pseudoinverse;
mod linear_sketch;
mod linalg_utils;
mod vector_space;
mod feature_collection;
mod sketched_linear_feature_collection;
mod embedder_state;
mod count_sketch;
mod quadratic_feature_collection;
mod closest_psd_matrix;
mod fourier_feature_collection;
mod func_scatter_tensor;
mod cauchy_fourier_features;
mod enum_feature_collection;
mod model;
mod inverse_schmear;
mod func_schmear;
mod func_inverse_schmear;
mod schmear;
mod model_space;
mod type_id;
mod params;
mod application_table;
mod interpreter_state;
mod func_impl;
mod term;
mod term_pointer;
mod term_reference;
mod term_application_result;
mod term_application;
mod type_space;
mod optimizer_state;
mod displayable_with_state;

extern crate lazy_static;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate pretty_env_logger;
#[macro_use] extern crate log;

use ndarray::*;
use ndarray_linalg::*;
use std::rc::*;

use crate::feature_collection::*;
use crate::displayable_with_state::*;
use crate::term_pointer::*;
use crate::inverse_schmear::*;
use crate::model::*;
use rand::prelude::*;
use crate::optimizer_state::*;

fn f(x : f32) -> f32 {
    2.0 * x
}

fn main() {
    pretty_env_logger::init();

    let num_iters = 40;
    let num_samples = 100;
    let in_dimensions = 1;
    let out_dimensions = 1;

    let mut rng = rand::thread_rng();

    let mut data_points : Vec::<(Array1<f32>, Array1<f32>)> = Vec::new();

    for i in 0..num_samples {
        let x : f32 = rng.gen();
        let y = f(x);

        let mut in_vec = Array::zeros((1,));
        in_vec[[0,]] = x;

        let mut out_vec = Array::zeros((1,));
        out_vec[[0,]] = y;
        
        let tuple : (Array1<f32>, Array1<f32>) = (in_vec, out_vec);

        data_points.push(tuple);
    }

    info!("Creating optimizer");
    let mut optimizer_state = OptimizerStateWithTarget::new(data_points);
    info!("Initializing optimizer");
    optimizer_state.init_step();
    info!("Running optimizer");
    for i in 0..num_iters {
        info!("Iter: {}", i);
        let term_ptr : TermPointer = optimizer_state.step();
        let term_str : String = term_ptr.display(&optimizer_state.optimizer_state.interpreter_state);
        info!("{}", term_str);
    }
}

