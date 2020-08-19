extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;
use std::rc::*;

use enum_dispatch::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::sketched_linear_feature_collection::*;
use crate::bayes_utils::*;

#[enum_dispatch(FeatureCollection)]
pub enum EnumFeatureCollection {
    SketchedLinearFeatureCollection,
    QuadraticFeatureCollection,
    FourierFeatureCollection
}

pub fn get_feature_collections(in_dimensions : usize) -> [EnumFeatureCollection; 3] {
    let linear_collection = SketchedLinearFeatureCollection::new(in_dimensions);
    let quadratic_collection = QuadraticFeatureCollection::new(in_dimensions);
    let fourier_collection = FourierFeatureCollection::new(in_dimensions, gen_cauchy_random);
    let feature_collections = [EnumFeatureCollection::from(linear_collection),
                               EnumFeatureCollection::from(quadratic_collection),
                               EnumFeatureCollection::from(fourier_collection)];
    feature_collections
}

pub fn get_rc_feature_collections(in_dimensions : usize) -> Rc<[EnumFeatureCollection; 3]> {
    let feature_collections = get_feature_collections(in_dimensions);
    Rc::new(feature_collections)
}

pub fn get_total_feat_dims(rc_feature_collections : &Rc<[EnumFeatureCollection; 3]>) -> usize {
    let mut total_feat_dims : usize = 0;
    for collection in rc_feature_collections.iter() {
        total_feat_dims += collection.get_dimension();
    }
    total_feat_dims
}
