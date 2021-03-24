extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::space_info::*;
use crate::term_model::*;
use crate::type_id::*;
use crate::data_point::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::params::*;
use rand::prelude::*;
use ndarray_linalg::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use crate::func_scatter_tensor::*;
use plotlib::page::Page;
use plotlib::repr::{Histogram, HistogramBins};
use plotlib::style::BoxStyle;
use plotlib::view::ContinuousView;
use crate::model::*;
use crate::normal_inverse_wishart::*;
use crate::term_reference::*;
use crate::array_utils::*;

pub fn term_ref(in_array : Array1<f32>) -> TermReference {
    TermReference::VecRef(to_noisy(&in_array))
}

pub fn random_scalar() -> f32 {
    let mut rng = rand::thread_rng();
    let result : f32 = rng.gen();
    result
}

pub fn random_data_point(in_dimensions : usize, out_dimensions : usize) -> DataPoint {
    let in_vec = random_vector(in_dimensions);
    let out_vec = random_vector(out_dimensions);

    let mut rng = rand::thread_rng();
    let weight_sqrt : f32 = rng.gen();
    let weight = weight_sqrt * weight_sqrt;
    
    DataPoint {
        in_vec,
        out_vec,
        weight
    }
}

pub fn standard_normal_inverse_wishart(feature_dimensions : usize, out_dimensions : usize) -> NormalInverseWishart {
    let mean = Array::zeros((out_dimensions, feature_dimensions));
    
    let in_precision = Array::eye(feature_dimensions);
    let out_precision = Array::eye(out_dimensions);
    let little_v = (out_dimensions as f32) + 2.0f32;

    NormalInverseWishart::new(mean, in_precision, out_precision, little_v)
}

pub fn random_normal_inverse_wishart(feature_dimensions : usize, out_dimensions : usize) -> NormalInverseWishart {
    let mean = random_matrix(out_dimensions, feature_dimensions);
    let precision = random_psd_matrix(feature_dimensions);
    let big_v = random_psd_matrix(out_dimensions);
    let little_v = (out_dimensions as f32) + 4.0f32;

    NormalInverseWishart::new(mean, precision, big_v, little_v)
}
//Yields a pair of models (func, arg), of type (in_dimensions -> middle_dimensions) ->
//out_dimensions
pub fn random_model_app(func_type_id : TypeId) -> (Model, Model) {
    let arg_type_id = get_arg_type_id(func_type_id);
    let arg_model = random_model(arg_type_id);
    let func_model = random_model(func_type_id);
    (func_model, arg_model)
}

pub fn random_model(type_id : TypeId) -> Model {
    let arg_type_id = get_arg_type_id(type_id);
    let ret_type_id = get_ret_type_id(type_id);

    let prior_specification = TermModelPriorSpecification { };

    let mut result = Model::new(&prior_specification, arg_type_id, ret_type_id);
    let func_space_info = get_function_space_info(type_id);
    result.data = random_normal_inverse_wishart(func_space_info.get_feature_dimensions(), 
                                                func_space_info.get_output_dimensions());
    result 
}

pub fn assert_equal_schmears(one : &Schmear, two : &Schmear) {
    assert_equal_matrices(&one.covariance, &two.covariance);
    assert_equal_vectors(&one.mean, &two.mean);
}

pub fn assert_equal_inv_schmears(one : &InverseSchmear, two : &InverseSchmear) {
    assert_equal_matrices(&one.precision, &two.precision);
    assert_equal_vectors(&one.mean, &two.mean);
}

pub fn relative_frob_norm_error(actual : &Array2<f32>, expected : &Array2<f32>) -> f32 {
    let denominator = expected.opnorm_fro().unwrap();
    let diff = actual - expected;
    let diff_norm = diff.opnorm_fro().unwrap();
    diff_norm / denominator
}

pub fn are_equal_matrices_to_within(one : &Array2<f32>, two : &Array2<f32>, within : f32, print : bool) -> bool {
    let diff = one - two;
    let frob_norm = diff.opnorm_fro().unwrap();
    if (frob_norm > within) {
        if (print) {
            println!("Actual: {}", one);
            println!("Expected: {}", two);
            println!("Diff: {}", diff);
            println!("Frob norm: {}", frob_norm);
        }
        false
    } else {
        true
    }
}

pub fn assert_equal_distributions_to_within(one : &NormalInverseWishart, two : &NormalInverseWishart, within : f32) {
    assert_equal_matrices_to_within(&one.mean, &two.mean, within);
    assert_equal_matrices_to_within(&one.precision, &two.precision, within);
    assert_equal_matrices_to_within(&one.big_v, &two.big_v, within);
    assert_eps_equals_to_within(one.little_v, two.little_v, within);
}

pub fn assert_equal_matrices_to_within(one : &Array2<f32>, two : &Array2<f32>, within : f32) {
    if (!are_equal_matrices_to_within(one, two, within, true)) {
        panic!();
    }
}

pub fn assert_equal_matrices(one : &Array2<f32>, two : &Array2<f32>) {
    assert_equal_matrices_to_within(one, two, DEFAULT_TEST_THRESH);
}

pub fn are_equal_vectors_to_within(one : &Array1<f32>, two : &Array1<f32>, within : f32, print : bool) -> bool {
    let diff = one - two;
    let sq_norm = diff.dot(&diff);
    let norm = sq_norm.sqrt();
    if (norm > within) {
        if (print) {
            println!("Actual: {}", one);
            println!("Expected: {}", two);
            println!("Diff: {}", diff);
            println!("Norm: {}", norm);
        }
        false
    } else {
        true
    }
}

pub fn assert_equal_vectors_to_within(one : &Array1<f32>, two : &Array1<f32>, within : f32) {
    if(!are_equal_vectors_to_within(one, two, within, true)) {
        panic!();
    }
}

pub fn assert_equal_vector_term(actual : TermReference, expected : Array1<f32>) {
    if let TermReference::VecRef(vec) = actual {
        assert_equal_vectors(&from_noisy(&vec), &expected);
    } else {
        panic!();
    }
}

pub fn assert_equal_vectors(one : &Array1<f32>, two : &Array1<f32>) {
    assert_equal_vectors_to_within(one, two, DEFAULT_TEST_THRESH);
}

pub fn assert_eps_equals_to_within(one : f32, two : f32, epsilon : f32) {
    let diff = one - two;
    if (diff.abs() > epsilon) {
        println!("Actual: {} Expected: {}", one, two);
        panic!();
    }
}

pub fn assert_eps_equals(one : f32, two : f32) {
    assert_eps_equals_to_within(one, two, DEFAULT_TEST_THRESH);
}
pub fn assert_greater(one : f32, two : f32) {
    if (two >= one) {
        println!("{} is greater than {}", two, one);
        panic!();
    }
}
pub fn random_vector(t : usize) -> Array1<f32> {
    Array::random((t,), StandardNormal)
}
pub fn random_matrix(t : usize, s : usize) -> Array2<f32> {
    Array::random((t, s), StandardNormal)
}
pub fn random_diag_matrix(t : usize) -> Array2<f32> {
    let mut result = Array::zeros((t, t));
    let diag = Array::random((t,), StandardNormal);
    for i in 0..t {
        result[[i, i]] = diag[[i,]];
    }
    result
}
pub fn random_psd_matrix(t : usize) -> Array2<f32> {
    let mut result = Array::zeros((t, t));
    for _ in 0..t {
        let matrix_sqrt = random_matrix(t, t);
        let matrix = matrix_sqrt.t().dot(&matrix_sqrt);
        result += &matrix;
    }
    result
}

pub fn random_schmear(t : usize) -> Schmear {
    let covariance = random_psd_matrix(t);
    let mean = random_vector(t);
    Schmear {
        mean,
        covariance
    }
}
pub fn random_inv_schmear(t : usize) -> InverseSchmear {
    let precision = random_psd_matrix(t);
    let mean = random_vector(t);
    InverseSchmear {
        mean,
        precision
    }
}

pub fn random_func_scatter_tensor(t : usize, s : usize) -> FuncScatterTensor {
    let in_mat = random_psd_matrix(s);
    let out_mat = random_psd_matrix(t);
    FuncScatterTensor::from_in_and_out_scatter(in_mat, out_mat)
}

pub fn random_func_schmear(t : usize, s : usize) -> FuncSchmear {
    let scatter_tensor = random_func_scatter_tensor(t, s);
    let mean = random_matrix(t, s);
    let result = FuncSchmear {
        mean : mean,
        covariance : scatter_tensor
    };
    result
}

pub fn empirical_gradient<F>(f : F, x : &Array1<f32>) -> Array1<f32>
    where F : Fn(&Array1<f32>) -> f32 {

    let epsilon = 0.001f32;
    let y = f(x);

    let s = x.shape()[0];

    let mut result = Array::zeros((s,));
    for i in 0..s {
        let mut delta_x : Array1<f32> = Array::zeros((s,));
        delta_x[[i,]] = epsilon;

        let new_x = x + &delta_x;
        let new_y = f(&new_x);
        let delta_y = new_y - y;

        let grad = delta_y / epsilon;

        result[[i,]] = grad;
    }
    result
}

pub fn empirical_jacobian<F>(f : F, x : &Array1<f32>) -> Array2<f32> 
    where F : Fn(&Array1<f32>) -> Array1<f32> {
    let epsilon = 0.001f32;
    let y = f(x);
    let s = x.shape()[0];
    let t = y.shape()[0];

    let mut result = Array::zeros((t, s));
    for i in 0..s {
        let mut delta_x : Array1<f32> = Array::zeros((s,));
        delta_x[[i,]] = epsilon;

        let new_x = x + &delta_x;
        let new_y = f(&new_x); 
        let delta_y = &new_y - &y;

        let grad = delta_y / epsilon;
        for j in 0..t {
            result[[j, i]] = grad[[j,]];
        }
    }
    result
}

pub fn plot_histogram(filename : &str, values : Vec<f64>, num_buckets : usize) {
    let h = Histogram::from_slice(&values, HistogramBins::Count(num_buckets))
            .style(&BoxStyle::new().fill("burlywood"));
    let v = ContinuousView::new().add(h);
    Page::single(&v).save("charts/".to_owned() + &filename + ".svg").expect("saving svg");
}

