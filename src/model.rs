extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use std::ops;
use std::rc::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::bayes_utils::*;
use arraymap::ArrayMap;

use std::collections::HashMap;

type UpdateKey = usize;

pub struct Model {
    in_dimensions : usize,
    out_dimensions : usize,
    feature_collections : Rc<[EnumFeatureCollection; 3]>,
    data : NormalInverseGamma,
    updates : HashMap::<UpdateKey, NormalInverseGamma>
}

pub fn to_features(feature_collections : &[EnumFeatureCollection; 3], in_vec : &Array1<f32>) -> Array1<f32> {
    let comps = feature_collections.map(|coll| coll.get_features(in_vec));
    stack(Axis(0), &[comps[0].view(), comps[1].view(), comps[2].view()]).unwrap()
}

pub fn to_jacobian(feature_collections : &[EnumFeatureCollection; 3], in_vec : &Array1<f32>) -> Array2<f32> {
    let comps = feature_collections.map(|coll| coll.get_jacobian(in_vec));
    stack(Axis(0), &[comps[0].view(), comps[1].view(), comps[2].view()]).unwrap()
}


impl Model {
    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        to_features(&self.feature_collections, in_vec)
    }

    fn get_data(&self, in_data : DataPoint) -> DataPoint {
        let feat_vec = self.get_features(&in_data.in_vec);

        DataPoint {
            in_vec : feat_vec,
            ..in_data
        }
    }

    pub fn eval(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let feats : Array1<f32> = self.get_features(in_vec);

        self.data.eval(&feats)
    }
}

impl ops::AddAssign<DataPoint> for Model {
    fn add_assign(&mut self, other: DataPoint) {
        self.data += &self.get_data(other);
    }
}

impl ops::SubAssign<DataPoint> for Model {
    fn sub_assign(&mut self, other: DataPoint) {
        self.data -= &self.get_data(other);
    }
}

impl Model {
    fn update_distr(&mut self, update_key : UpdateKey, distr : NormalInverseGamma) {
        self.data += &distr;
        self.updates.insert(update_key, distr);
    }
    fn downdate_distr(&mut self, key : &UpdateKey) {
        let mut distr = self.updates.remove(key).unwrap();
        distr ^= ();
        self.data += &distr;
    }
}

impl Model {
    pub fn new(feature_collections : Rc<[EnumFeatureCollection; 3]>,
              in_dimensions : usize, out_dimensions : usize) -> Model {

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

                let precision_block = if i == j {
                    collection_i.blank_diagonal_precision(out_dimensions)
                } else {
                    collection_i.blank_interaction_precision(collection_j, out_dimensions)
                };

                for k in 0..coll_i_size {
                    for l in 0..coll_j_size {
                        let k_offset = ind_one + k;
                        let l_offset = ind_two + l;

                        for t_one in 0..out_dimensions {
                            for t_two in 0..out_dimensions {
                                precision[[t_one, k_offset, t_two, l_offset]] = precision_block[[t_one, k, t_two, l]];
                            }
                        }
                    }
                }
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
            updates
        }
    }
}


