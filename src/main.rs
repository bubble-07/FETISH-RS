#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod newly_evaluated_terms;
mod input_to_schmeared_output;
mod nonprimitive_term_pointer;
mod primitive_term_pointer;
mod term_index;
mod displayable_with_context;
mod primitive_type_space;
mod primitive_directory;
mod context;
mod compressed_inv_schmear;
mod sampled_value_field_state;
mod sampled_value_field;
mod prior_specification;
mod elaborator;
mod kernel;
mod space_info;
mod type_graph;
mod type_action;
mod application_chain;
mod interpreter_and_embedder_state;
mod constraint_collection;
mod vector_application_result;
mod function_optimum_space;
mod function_optimum_state;
mod value_field_maximum_solver;
mod typed_vector;
mod value_field_state;
mod value_field;
mod feature_space_info;
mod function_space_info;
mod schmeared_hole;
mod schmear_sampler;
mod sampled_model_embedding;
mod sqrtm;
mod data_points;
mod rand_utils;
mod term_model;
mod sampled_embedder_state;
mod sampled_embedding_space;
mod sigma_points;
mod sherman_morrison;
mod data_point;
mod normal_inverse_wishart;
mod normal_inverse_wishart_sampler;
mod wishart;
mod alpha_formulas;
mod test_utils;
mod array_utils;
mod pseudoinverse;
mod linear_sketch;
mod linalg_utils;
mod feature_collection;
mod sketched_linear_feature_collection;
mod embedder_state;
mod count_sketch;
mod quadratic_feature_collection;
mod fourier_feature_collection;
mod func_scatter_tensor;
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

extern crate ndarray;
extern crate ndarray_linalg;
extern crate pretty_env_logger;
#[macro_use] extern crate log;

use ndarray::*;
use crate::displayable_with_state::*;
use rand::prelude::*;
use crate::optimizer_state::*;
use crate::context::*;

fn f(x : f32) -> f32 {
    2.0 * x + 1.0
}

fn main() {
    pretty_env_logger::init();

    let num_iters = 40;
    let num_samples = 100;

    let mut rng = rand::thread_rng();

    let mut data_points : Vec::<(Array1<f32>, Array1<f32>)> = Vec::new();

    for _ in 0..num_samples {
        let x : f32 = rng.gen();
        let y = f(x);

        let mut in_vec = Array::zeros((1,));
        in_vec[[0,]] = x;

        let mut out_vec = Array::zeros((1,));
        out_vec[[0,]] = y;
        
        let tuple : (Array1<f32>, Array1<f32>) = (in_vec, out_vec);

        data_points.push(tuple);
    }

    info!("Building context");
    let context = get_default_context();

    info!("Creating optimizer");
    let mut optimizer_state = OptimizerState::new(data_points, &context);
    info!("Initializing optimizer");
    optimizer_state.init_step();
    info!("Running optimizer");
    for i in 0..num_iters {
        info!("Iter: {}", i);
        let maybe_term_ptr = optimizer_state.step();
        if let Option::Some(term_ptr) = maybe_term_ptr {
            let term_str : String = term_ptr.display(&optimizer_state.interpreter_and_embedder_state.interpreter_state);
            info!("{}", term_str);
        } else {
            info!("No term from optimizer");
        }
    }
}

