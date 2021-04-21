#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod params;

mod alpha_formulas;
mod context;
mod elaborator;
mod feature_collection;
mod feature_space_info;
mod space_info;
mod interpreter_and_embedder_state;
mod interpreter_state;
mod sampled_embedder_state;
mod sampled_embedding_space;
mod term_model;

mod sampled_value_field_state;
mod sampled_value_field;
mod type_graph;
mod type_action;
mod application_chain;
mod constraint_collection;
mod vector_application_result;
mod function_optimum_space;
mod function_optimum_state;
mod value_field_maximum_solver;
mod value_field_state;
mod value_field;
mod schmear_sampler;
mod optimizer_state;

extern crate ndarray;
extern crate ndarray_linalg;
extern crate pretty_env_logger;
extern crate fetish_lib;
#[macro_use] extern crate log;

use ndarray::*;
use rand::prelude::*;
use crate::optimizer_state::*;
use crate::context::*;
use crate::elaborator::*;
use crate::term_model::*;
use fetish_lib::everything::*;

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

    let elaborator_prior_specification = ElaboratorPrior { };
    let model_prior_specification = TermModelPriorSpecification { };

    info!("Building context");
    let context = get_default_context();

    info!("Creating optimizer");
    let mut optimizer_state = OptimizerState::new(data_points, &model_prior_specification, 
                                                  &elaborator_prior_specification, &context);
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

