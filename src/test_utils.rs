extern crate ndarray;
extern crate ndarray_linalg;

use std::collections::HashMap;
use ndarray::*;
use crate::context::*;
use crate::space_info::*;
use crate::term_model::*;
use crate::type_id::*;
use crate::prior_info::*;
use crate::prior_directory::*;
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
use crate::model::*;
use crate::normal_inverse_wishart::*;
use crate::term_reference::*;
use crate::prior_specification::*;
use crate::array_utils::*;
use crate::feature_space_info::*;
use crate::feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::sketched_linear_feature_collection::*;
use crate::primitive_directory::*;
use crate::rand_utils::*;

///A collection of crate-internal utilities for constructing tests.

pub const TEST_VECTOR_T : TypeId = 1 as TypeId;
pub const TEST_SCALAR_T : TypeId = 0 as TypeId;
pub const TEST_VECTOR_SIZE : usize = 2;

fn get_test_vector_only_type_info_directory() -> TypeInfoDirectory {
    let mut result = TypeInfoDirectory::new();
    result.add(Type::VecType(1));
    result.add(Type::VecType(TEST_VECTOR_SIZE));
    result
}

pub fn get_test_vector_only_feature_space_info(base_dimensions : usize) -> FeatureSpaceInfo {
    let sketcher = Option::None; 
    let mut feature_collections = Vec::<Box<dyn FeatureCollection>>::new();
    let fourier_feature_collection = FourierFeatureCollection::new(base_dimensions, base_dimensions * 2, 1.0f32,
                                                          gen_nsphere_random);
    let sketched_linear_feature_collection = SketchedLinearFeatureCollection::new(base_dimensions, base_dimensions * 2, 1.0f32);
    feature_collections.push(Box::new(fourier_feature_collection));
    feature_collections.push(Box::new(sketched_linear_feature_collection));

    let feature_dimensions = get_total_feat_dims(&feature_collections);

    FeatureSpaceInfo {
        base_dimensions,
        feature_dimensions,
        feature_collections,
        sketcher
    }
}

fn get_test_vector_only_space_info_directory() -> SpaceInfoDirectory {
    let mut feature_spaces = Vec::new();

    feature_spaces.push(get_test_vector_only_feature_space_info(1));
    feature_spaces.push(get_test_vector_only_feature_space_info(TEST_VECTOR_SIZE));

    SpaceInfoDirectory {
        feature_spaces
    }
}

fn get_test_vector_only_prior_info_directory() -> PriorDirectory {
    //Empty, because we have no function types
    let priors = HashMap::new();
    PriorDirectory {
        priors
    }
}

pub fn get_test_vector_only_context() -> Context {
    let type_info_directory = get_test_vector_only_type_info_directory();
    let space_info_directory = get_test_vector_only_space_info_directory();
    let primitive_directory = PrimitiveDirectory::new(&type_info_directory);
    let prior_directory = get_test_vector_only_prior_info_directory();
    Context {
        type_info_directory,
        space_info_directory,
        primitive_directory,
        prior_directory
    }
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
pub fn random_model_app<'a>(ctxt : &'a Context, func_type_id : TypeId) -> (Model<'a>, Model<'a>) {
    let arg_type_id = ctxt.get_arg_type_id(func_type_id);

    let in_type_id = ctxt.get_arg_type_id(arg_type_id);
    let middle_type_id = ctxt.get_ret_type_id(arg_type_id);

    let ret_type_id = ctxt.get_ret_type_id(func_type_id);
    let arg_model = random_model(ctxt, in_type_id, middle_type_id);
    let func_model = random_model(ctxt, arg_type_id, ret_type_id);
    (func_model, arg_model)
}

pub struct TestPriorSpecification { }
impl PriorSpecification for TestPriorSpecification {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        1.0f32
    }
    fn get_out_covariance_multiplier(&self, _out_dims : usize) -> f32 {
        1.0f32
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        (out_dims as f32) + 4.0f32
    }
}

pub fn random_model<'a>(ctxt : &'a Context, arg_type_id : TypeId, ret_type_id : TypeId) -> Model {
    let prior_specification = TestPriorSpecification { };

    let mut result = Model::new(&prior_specification, arg_type_id, ret_type_id, ctxt);
    let arg_feat_space_info = ctxt.get_feature_space_info(arg_type_id);
    let ret_feat_space_info = ctxt.get_feature_space_info(ret_type_id);
    result.data = random_normal_inverse_wishart(arg_feat_space_info.feature_dimensions, 
                                                ret_feat_space_info.base_dimensions);
    result 
}

pub fn assert_equal_schmears(one : &Schmear, two : &Schmear) {
    assert_equal_matrices(one.covariance.view(), two.covariance.view());
    assert_equal_vectors(one.mean.view(), two.mean.view());
}

pub fn assert_equal_inv_schmears(one : &InverseSchmear, two : &InverseSchmear) {
    assert_equal_matrices(one.precision.view(), two.precision.view());
    assert_equal_vectors(one.mean.view(), two.mean.view());
}

pub fn relative_frob_norm_error(actual : ArrayView2<f32>, expected : ArrayView2<f32>) -> f32 {
    let denominator = expected.opnorm_fro().unwrap();
    let diff = &actual - &expected;
    let diff_norm = diff.opnorm_fro().unwrap();
    diff_norm / denominator
}

pub fn are_equal_matrices_to_within(one : ArrayView2<f32>, two : ArrayView2<f32>, within : f32, print : bool) -> bool {
    let diff = &one - &two;
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
    assert_equal_matrices_to_within(one.mean.view(), two.mean.view(), within);
    assert_equal_matrices_to_within(one.precision.view(), two.precision.view(), within);
    assert_equal_matrices_to_within(one.big_v.view(), two.big_v.view(), within);
    assert_eps_equals_to_within(one.little_v, two.little_v, within);
}

pub fn assert_equal_matrices_to_within(one : ArrayView2<f32>, two : ArrayView2<f32>, within : f32) {
    if (!are_equal_matrices_to_within(one, two, within, true)) {
        panic!();
    }
}

pub fn assert_equal_matrices(one : ArrayView2<f32>, two : ArrayView2<f32>) {
    assert_equal_matrices_to_within(one, two, DEFAULT_TEST_THRESH);
}

pub fn are_equal_vectors_to_within(one : ArrayView1<f32>, two : ArrayView1<f32>, within : f32, print : bool) -> bool {
    let diff = &one - &two;
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

pub fn assert_equal_vectors_to_within(one : ArrayView1<f32>, two : ArrayView1<f32>, within : f32) {
    if(!are_equal_vectors_to_within(one, two, within, true)) {
        panic!();
    }
}

pub fn assert_equal_vector_term(actual : TermReference, expected : ArrayView1<f32>) {
    if let TermReference::VecRef(_, vec) = actual {
        assert_equal_vectors(from_noisy(vec.view()).view(), expected);
    } else {
        panic!();
    }
}

pub fn assert_equal_vectors(one : ArrayView1<f32>, two : ArrayView1<f32>) {
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
    let in_scatter = random_psd_matrix(s);
    let out_scatter = random_psd_matrix(t);
    FuncScatterTensor {
        in_scatter,
        out_scatter
    }
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

pub fn empirical_gradient<F>(f : F, x : ArrayView1<f32>) -> Array1<f32>
    where F : Fn(ArrayView1<f32>) -> f32 {

    let epsilon = 0.001f32;
    let y = f(x);

    let s = x.shape()[0];

    let mut result = Array::zeros((s,));
    for i in 0..s {
        let mut delta_x : Array1<f32> = Array::zeros((s,));
        delta_x[[i,]] = epsilon;

        let new_x = &x + &delta_x;
        let new_y = f(new_x.view());
        let delta_y = new_y - y;

        let grad = delta_y / epsilon;

        result[[i,]] = grad;
    }
    result
}

pub fn empirical_jacobian<F>(f : F, x : ArrayView1<f32>) -> Array2<f32> 
    where F : Fn(ArrayView1<f32>) -> Array1<f32> {
    let epsilon = 0.001f32;
    let y = f(x);
    let s = x.shape()[0];
    let t = y.shape()[0];

    let mut result = Array::zeros((t, s));
    for i in 0..s {
        let mut delta_x : Array1<f32> = Array::zeros((s,));
        delta_x[[i,]] = epsilon;

        let new_x = &x + &delta_x;
        let new_y = f(new_x.view()); 
        let delta_y = &new_y - &y;

        let grad = delta_y / epsilon;
        for j in 0..t {
            result[[j, i]] = grad[[j,]];
        }
    }
    result
}
