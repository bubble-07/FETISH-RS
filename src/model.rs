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
use crate::term_application::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::sampled_function::*;
use arraymap::ArrayMap;
use rand::prelude::*;

use std::collections::HashMap;

type PriorUpdateKey = TermApplication;
type DataUpdateKey = TermReference;

pub struct Model {
    in_dimensions : usize,
    out_dimensions : usize,
    feature_collections : Rc<[EnumFeatureCollection; 3]>,
    data : NormalInverseGamma,
    prior_updates : HashMap::<PriorUpdateKey, NormalInverseGamma>,
    data_updates : HashMap::<DataUpdateKey, DataPoint>
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
    pub fn sample(&self, rng : &mut ThreadRng) -> SampledFunction {
        let mat = self.data.sample(rng);
        SampledFunction {
            in_dimensions : self.in_dimensions,
            mat : mat,
            feature_collections : self.feature_collections.clone()
        }
    }
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        self.data.sample_as_vec(rng)
    }
    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        self.data.get_mean_as_vec()
    }
    pub fn get_inverse_schmear(&self) -> InverseSchmear {
        self.data.get_inverse_schmear()
    }

    pub fn get_schmear(&self) -> Schmear {
        self.data.get_schmear()
    }

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
    pub fn has_data(&self, update_key : &DataUpdateKey) -> bool {
        self.data_updates.contains_key(update_key)
    }
    pub fn update_data(&mut self, update_key : DataUpdateKey, data_point : DataPoint) {
        self.data += &data_point;
        self.data_updates.insert(update_key, data_point);
    }
    pub fn downdate_data(&mut self, update_key : &DataUpdateKey) {
        let added_point : DataPoint = self.data_updates.remove(update_key).unwrap();
        self.data -= &added_point;
    }
}

impl Model {
    pub fn has_prior(&self, update_key : &PriorUpdateKey) -> bool {
        self.prior_updates.contains_key(update_key)
    }
    pub fn update_prior(&mut self, update_key : PriorUpdateKey, distr : NormalInverseGamma) {
        self.data += &distr;
        self.prior_updates.insert(update_key, distr);
    }
    pub fn downdate_prior(&mut self, key : &PriorUpdateKey) {
        let mut distr = self.prior_updates.remove(key).unwrap();
        distr ^= ();
        self.data += &distr;
    }
}

impl Model {
    pub fn new(feature_collections : Rc<[EnumFeatureCollection; 3]>,
              in_dimensions : usize, out_dimensions : usize) -> Model {

        let prior_updates : HashMap::<PriorUpdateKey, NormalInverseGamma> = HashMap::new();
        let data_updates : HashMap::<DataUpdateKey, DataPoint> = HashMap::new();

        let mut total_feat_dims : usize = 0;
        for collection in feature_collections.iter() {
            total_feat_dims += collection.get_dimension();
        }

        let mut mean : Array2<f32> = Array::zeros((out_dimensions, total_feat_dims));
        let mut ind_one : usize = 0;

        for (i, collection_i) in feature_collections.iter().enumerate() {
            let coll_i_size : usize = collection_i.get_dimension();
            let end_ind_one = ind_one + coll_i_size;

            let mean_block : Array2<f32> = collection_i.blank_mean(out_dimensions);
            for t in 0..out_dimensions {
                for k in 0..coll_i_size {
                    let k_offset = ind_one + k;
                    mean[[t, k_offset]] = mean_block[[t, k]];
                }
            }

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
            prior_updates,
            data_updates
        }
    }
}


