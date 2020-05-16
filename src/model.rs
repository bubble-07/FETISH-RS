extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::bayes_utils::*;

use std::collections::HashMap;

struct Model {
    in_dimensions : usize,
    out_dimensions : usize,
    feature_collections : [EnumFeatureCollection; 3],
    data : NormalInverseGamma,
    update_key_counter : usize,
    updates : HashMap::<usize, NormalInverseGamma>
}

impl Model {
    fn new(in_dimensions : usize, out_dimensions : usize) -> Model {
        let linear_collection = LinearFeatureCollection::new(in_dimensions);
        let quadratic_collection = QuadraticFeatureCollection::new(in_dimensions);
        let fourier_collection = FourierFeatureCollection::new(in_dimensions, gen_cauchy_random);
        let feature_collections = [EnumFeatureCollection::from(linear_collection),
                                   EnumFeatureCollection::from(quadratic_collection),
                                   EnumFeatureCollection::from(fourier_collection)];

        let update_key_counter : usize = 0; 
        let updates : HashMap::<usize, NormalInverseGamma> = HashMap::new();

        let mut total_feat_dims : usize = 0;
        for collection in feature_collections.iter() {
            total_feat_dims += collection.get_dimension();
        }

        let mut mean : Array2<f32> = Array::zeros((out_dimensions, total_feat_dims));
        let mut ind_one : usize = 0;

        for collection in feature_collections.iter() {
            let coll_size : usize = collection.get_dimension();
            let end_ind_one = ind_one + coll_size;
            let mut mean_slice = mean.slice_mut(s![..,ind_one..end_ind_one]);
            let mut mean_block : Array2<f32> = collection.blank_mean(out_dimensions);
            mean_slice = mean_block.view_mut();
            ind_one = end_ind_one;
        }

        let mut precision : Array4<f32> = Array::zeros((out_dimensions, total_feat_dims, out_dimensions, total_feat_dims));
        let mut ind_one = 0;

        for (i, collection_i) in feature_collections.iter().enumerate() {

            let coll_i_size : usize = collection_i.get_dimension();
            let end_ind_one = ind_one + coll_i_size;

            let mut ind_two : usize = 0;

            for (j, collection_j) in feature_collections.iter().enumerate() {
                let coll_j_size : usize = collection_j.get_dimension();
                let end_ind_two = ind_two + coll_j_size; 

                let mut precision_block = if (i == j) {
                    collection_i.blank_diagonal_precision(out_dimensions)
                } else {
                    collection_i.blank_interaction_precision(collection_j, out_dimensions)
                };

                let mut precision_slice = precision.slice_mut(s![..,ind_one..end_ind_one,..,ind_two..end_ind_two]);
                precision_slice = precision_block.view_mut();

                ind_two = end_ind_two;
            }

            ind_one = end_ind_one;
        }

        let data = NormalInverseGamma::new(mean, precision, 0.5, 0.0, out_dimensions, in_dimensions);
    
        Model {
            in_dimensions,
            out_dimensions,
            feature_collections,
            data,
            update_key_counter,
            updates
        }
    }
}


