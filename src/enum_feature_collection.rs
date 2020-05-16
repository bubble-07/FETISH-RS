extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use enum_dispatch::*;
use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::bayes_utils::*;

#[enum_dispatch(FeatureCollection)]
pub enum EnumFeatureCollection {
    LinearFeatureCollection,
    QuadraticFeatureCollection,
    FourierFeatureCollection
}
