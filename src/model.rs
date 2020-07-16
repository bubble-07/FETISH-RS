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
use crate::normal_inverse_wishart::*;
use crate::term_application::*;
use crate::term_pointer::*;
use crate::params::*;
use crate::term_reference::*;
use crate::schmear::*;
use crate::data_point::*;
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
    data : NormalInverseWishart,
    prior_updates : HashMap::<PriorUpdateKey, NormalInverseWishart>,
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

    //Find a better function and a better argument in the case where both
    //have schmears
    pub fn find_better_app(&self, arg : &Model, target : &Array1<f32>) -> (InverseSchmear, InverseSchmear) {
        let func_inv_schmear = self.data.get_inverse_schmear();
        let arg_inv_schmear = arg.data.get_inverse_schmear();
        let u_x : Array1<f32> = arg_inv_schmear.mean;
        let p_x : Array2<f32> = arg_inv_schmear.precision;
        let u_f : Array2<f32> = self.data.get_mean();

        //t x z x t x z
        let p_f : Array4<f32> = self.data.get_precision();

        //z
        let k = self.get_features(&u_x);
        let k_t_k = einsum("a,a->", &[&k, &k]).unwrap()
                    .into_dimensionality::<Ix0>().unwrap().into_scalar();
        let alpha = 1.0f32 / k_t_k;

        let u_f_k : Array1<f32> = einsum("ab,b->a", &[&u_f, &k]).unwrap()
                                  .into_dimensionality::<Ix1>().unwrap();
        //t
        let r = target - &u_f_k;
        //z x s
        let J = to_jacobian(&self.feature_collections, &u_x);

        //t x z
        let mut a : Array2<f32> = einsum("a,b->ab", &[&r, &k]).unwrap()
                              .into_dimensionality::<Ix2>().unwrap();

        a *= alpha;

        //t x s
        let u_f_J : Array2<f32> = einsum("ab,bc->ac", &[&u_f, &J]).unwrap()
                              .into_dimensionality::<Ix2>().unwrap();

        //t x z x s
        let J_r : Array3<f32> = einsum("zs,t->tzs", &[&J, &r]).unwrap()
                              .into_dimensionality::<Ix3>().unwrap();

        //t x z x s
        let k_u_f_J : Array3<f32> = einsum("z,ts->tzs", &[&k, &u_f_J]).unwrap()
                              .into_dimensionality::<Ix3>().unwrap();

        
        //t x z x s
        let B : Array3<f32> = alpha * (J_r - k_u_f_J);

        //t x z x s
        let B_t_p_f : Array3<f32> = einsum("tzs,tzab->abs", &[&B, &p_f]).unwrap()
                              .into_dimensionality::<Ix3>().unwrap();
        
        //s x s
        let B_t_p_f_B : Array2<f32> = einsum("tza,tzb->ab", &[&B_t_p_f, &B]).unwrap()
                              .into_dimensionality::<Ix2>().unwrap();

        //B_t_p_f_B + p_x : s x s
        let inner : Array2<f32> = B_t_p_f_B + &p_x;
        let inner_inv = inner.invh().unwrap();


        //inner_inv : s x s,
        //B_t_p_f : t x z x s
        //a : t x z
        let inner_inv_B_t_p_f_a : Array1<f32> = einsum("os,tzs,tz->o", &[&inner_inv, &B_t_p_f, &a]).unwrap()
                               .into_dimensionality::<Ix1>().unwrap();

        let delta_x : Array1<f32> = -inner_inv_B_t_p_f_a;
         
        //Now that we have estimated what the change in x should
        //be [under the linear approximation by the jacobian]
        //we just need to find the smallest corresponding
        //change in f which exactly makes (u_f + d_f)(u_x + d_x) = y
        
        let new_x : Array1<f32> = u_x + delta_x;
        let new_k : Array1<f32> = self.get_features(&new_x);

        let new_k_sq_norm : f32 = einsum("a,a->", &[&new_k, &new_k]).unwrap()
                                  .into_dimensionality::<Ix0>().unwrap().into_scalar();

        let u_f_new_k : Array1<f32> = einsum("ab,b->a", &[&u_f, &new_k]).unwrap()
                                      .into_dimensionality::<Ix1>().unwrap();

        let norm_new_k : Array1<f32> = (1.0f32 / new_k_sq_norm) * new_k;

        
        let t : Array1<f32> = target - &u_f_new_k;

        let delta_f : Array2<f32> = einsum("t,s->ts", &[&t, &norm_new_k]).unwrap()
                                      .into_dimensionality::<Ix2>().unwrap();

        let new_f = u_f + delta_f;

        let result_f = InverseSchmear {
            mean : mean_to_array(&new_f),
            precision : func_inv_schmear.precision.clone()
        };
        
        let result_x = InverseSchmear {
            mean : new_x,
            precision : p_x
        };

        (result_f, result_x) 
    }

    //Find a better function in the case where the argument is a vector
    pub fn find_better_func(&self, arg : &Array1<f32>, target : &Array1<f32>) -> InverseSchmear {
        let k = self.get_features(arg);
        let k_t_k = einsum("a,a->", &[&k, &k]).unwrap()
                          .into_dimensionality::<Ix0>().unwrap().into_scalar();
        let k_normed = k * (1.0f32 / k_t_k);

        let func_inv_schmear = self.data.get_inverse_schmear();
        let new_out = self.eval(arg);
        let r = target - &new_out;

        let delta_f : Array2<f32> = einsum("t,s->ts", &[&r, &k_normed]).unwrap()
                                    .into_dimensionality::<Ix2>().unwrap();
        let new_f = func_inv_schmear.mean + mean_to_array(&delta_f);

        InverseSchmear {
            mean : new_f,
            precision : func_inv_schmear.precision.clone()
        }
    }
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
    pub fn update_prior(&mut self, update_key : PriorUpdateKey, distr : NormalInverseWishart) {
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

        println!("Initializing model with dims {} -> {}", in_dimensions, out_dimensions);

        let prior_updates : HashMap::<PriorUpdateKey, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<DataUpdateKey, DataPoint> = HashMap::new();

        let mut total_feat_dims : usize = 0;
        for collection in feature_collections.iter() {
            total_feat_dims += collection.get_dimension();
        }

        println!("Initializing model mean");

        let mut mean : Array2<f32> = Array::zeros((out_dimensions, total_feat_dims));
        let mut ind_one : usize = 0;

        for (i, collection_i) in feature_collections.iter().enumerate() {
            let coll_i_size : usize = collection_i.get_dimension();
            let end_ind_one = ind_one + coll_i_size;

            let mean_block : Array2<f32> = collection_i.blank_mean(out_dimensions);

            mean.slice_mut(s![.., ind_one..end_ind_one]).assign(&mean_block);

            ind_one = end_ind_one;
        }

        println!("Initializing model precision");

        let mut in_precision : Array2<f32> = Array::zeros((total_feat_dims, total_feat_dims));
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

                in_precision.slice_mut(s![ind_one..end_ind_one, ind_two..end_ind_two])
                         .assign(&precision_block);

                ind_two = end_ind_two;
            }
            ind_one = end_ind_one;
        }

        let out_precision = (out_dimensions as f32) * OUT_REG_STRENGTH * Array::eye(out_dimensions);

        let little_v = out_dimensions as f32;

        println!("Initializing model initial distribution");

        let data = NormalInverseWishart::new(mean, in_precision, out_precision, little_v);
    
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


