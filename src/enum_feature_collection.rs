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

pub fn get_feature_collections(in_dimensions : usize) -> [EnumFeatureCollection; 3] {
    let linear_collection = LinearFeatureCollection::new(in_dimensions);
    let quadratic_collection = QuadraticFeatureCollection::new(in_dimensions);
    let fourier_collection = FourierFeatureCollection::new(in_dimensions, gen_cauchy_random);
    let feature_collections = [EnumFeatureCollection::from(linear_collection),
                               EnumFeatureCollection::from(quadratic_collection),
                               EnumFeatureCollection::from(fourier_collection)];
    feature_collections
}
