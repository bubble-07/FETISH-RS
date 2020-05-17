mod bayes_utils;
mod linalg_utils;
mod feature_collection;
mod linear_feature_collection;
mod count_sketch;
mod quadratic_feature_collection;
mod fourier_feature_collection;
mod cauchy_fourier_features;
mod enum_feature_collection;
mod model;

extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;


use crate::model::*;
use plotters::prelude::*;
use rand::prelude::*;
use bayes_utils::*;

fn f(x : f32) -> f32 {
    -x * x
}

fn main() {
    let num_samples = 100;

    let mut model : Model = Model::new(1, 1);
    let mut rng = rand::thread_rng();

    for i in 0..num_samples {
        let mut x : f32 = rng.gen();
        let mut noise : f32 = rng.gen();
        let y = f(x) + (noise - 0.5) * 0.1;

        let out_precision = Array::ones((1,1));

        let mut in_vec = Array::zeros((1,));
        in_vec[[0,]] = x;

        let mut out_vec = Array::zeros((1,));
        out_vec[[0,]] = y;

        let data_point = DataPoint {
            in_vec,
            out_vec,
            out_precision
        };
        model += data_point;

        let out_precision = Array::ones((1,1));

        let mut in_vec = Array::zeros((1,));
        in_vec[[0,]] = x;

        let mut out_vec = Array::zeros((1,));
        out_vec[[0,]] = y;

        let data_point = DataPoint {
            in_vec,
            out_vec,
            out_precision
        };
        model -= data_point;

    }

    fn model_fn(model : &Model, x : f64) -> f64 {
        let mut x_arr : Array1<f32> = Array::ones((1,));
        x_arr[[0,]] = x as f32;
        let y_arr = model.eval(&x_arr);
        y_arr[[0,]] as f64
    }

    let root_drawing_area = BitMapBackend::new("result.png", (1024, 768)).into_drawing_area();

    root_drawing_area.fill(&WHITE);

    let mut chart = ChartBuilder::on(&root_drawing_area)
                            .build_ranged(-1.0..1.0, -1.0..1.0)
                            .unwrap();
    chart.draw_series(LineSeries::new(
            (-100..100).map(|x| x as f64 / 100.0).map(|x| (x, model_fn(&model, x)) ), &RED
            )).unwrap();
}

