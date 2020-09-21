extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::data_update::*;
use crate::data_point::*;
use crate::pseudoinverse::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;
use crate::linalg_utils::*;
use crate::normal_inverse_wishart::*;
use crate::term_application::*;
use crate::func_scatter_tensor::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::params::*;
use crate::test_utils::*;

use crate::sampled_function::*;
use arraymap::ArrayMap;
use rand::prelude::*;

use std::collections::HashMap;

#[derive(Clone)]
pub struct Model {
    in_dimensions : usize,
    out_dimensions : usize,
    feature_collections : Rc<[EnumFeatureCollection; 3]>,
    pub data : NormalInverseWishart,
    prior_updates : HashMap::<TermApplication, NormalInverseWishart>,
    pub data_updates : HashMap::<TermReference, DataUpdate>
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

    pub fn get_total_dims(&self) -> usize {
        self.data.get_total_dims()
    }

    fn find_better_internal(&self, arg : InverseSchmear, target : &Array1<f32>) -> (InverseSchmear, InverseSchmear) {
        let func_inv_schmear = self.data.get_inverse_schmear();
        let u_x : Array1<f32> = arg.mean;
        let p_x : Array2<f32> = arg.precision;
        let u_f : Array2<f32> = self.data.get_mean();

        //t x z x t x z
        let p_f : FuncScatterTensor = self.data.get_precision();

        //z
        let k = self.get_features(&u_x);
        let k_t_k = k.dot(&k);


        let u_f_k = u_f.dot(&k);
        //t
        let r = target - &u_f_k;
        //z x s
        let J = to_jacobian(&self.feature_collections, &u_x);

        let u_f_J = u_f.dot(&J);

        let p_t = p_f.out_scatter;
        let p_s = p_f.scale * p_f.in_scatter;


        let J_u_p_t = u_f_J.t().dot(&p_t);

        let J_u_p_t_r = J_u_p_t.dot(&r);
        let J_u_p_t_u_J = J_u_p_t.dot(&u_f_J);

        let k_p_s_k : f32 = k.dot(&p_s).dot(&k);
        let p_x_scale = (k_t_k * k_t_k) / (k_p_s_k);
        let scaled_p_x = p_x_scale * p_x.clone();

        let inner = scaled_p_x + J_u_p_t_u_J;
        let inner_inv = pseudoinverse_h(&inner);

        let delta_x : Array1<f32> = inner_inv.dot(&J_u_p_t_r);
         
        //Now that we have estimated what the change in x should
        //be [under the linear approximation by the jacobian]
        //we just need to find the smallest corresponding
        //change in f which exactly makes (u_f + d_f)(u_x + d_x) = y
        
        let new_x : Array1<f32> = u_x + delta_x;
        let new_k : Array1<f32> = self.get_features(&new_x);

        let new_k_sq_norm = new_k.dot(&new_k);

        let u_f_new_k = u_f.dot(&new_k);

        let norm_new_k : Array1<f32> = (1.0f32 / new_k_sq_norm) * new_k;

        
        let t : Array1<f32> = target - &u_f_new_k;

        let delta_f : Array2<f32> = outer(&t, &norm_new_k);

        let new_f = u_f + delta_f;

        let result_f = InverseSchmear {
            mean : mean_to_array(&new_f),
            precision : func_inv_schmear.precision.flatten()
        };
        
        let result_x = InverseSchmear {
            mean : new_x,
            precision : p_x
        };

        (result_f, result_x) 

    }

    //Find a better function and a better argument in the case where the
    //argument is a vector
    pub fn find_better_vec_app(&self, arg : &Array1<f32>, target : &Array1<f32>) -> (InverseSchmear, Array1<f32>) {
        let arg_inv_schmear = InverseSchmear::zero_precision_from_vec(arg);

        let (better_func_schmear, better_arg_schmear) = self.find_better_internal(arg_inv_schmear, target);
        let better_vec = better_arg_schmear.mean;
        (better_func_schmear, better_vec)
    }

    //Find a better function and a better argument in the case where both
    //have schmears
    pub fn find_better_app(&self, arg_schmear : InverseSchmear, target : &Array1<f32>) -> (InverseSchmear, InverseSchmear) {
        self.find_better_internal(arg_schmear, target)
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

    pub fn get_single_observation_weight(&self) -> f32 {
        let in_weight = self.data.precision.opnorm_fro().unwrap();
        let out_weight = self.data.big_v.opnorm_fro().unwrap();
        let combined_weight = (in_weight * out_weight).sqrt();

        let num_observations = self.data.little_v - (self.data.t as f32);

        let result = combined_weight / num_observations;
        
        if (result < 0.0f32) {
            println!("Single obs weight below zero: {}", result);
        }
        result
    }

    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        self.data.get_inverse_schmear()
    }

    pub fn get_schmear(&self) -> FuncSchmear {
        self.data.get_schmear()
    }

    pub fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
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
    pub fn has_data(&self, update_key : &TermReference) -> bool {
        self.data_updates.contains_key(update_key)
    }
    pub fn update_data(&mut self, update_key : TermReference, data_update : DataUpdate) {
        let feat_update = data_update.featurize(&self);
        self.data += &feat_update;
        self.data_updates.insert(update_key, feat_update);
    }
    pub fn downdate_data(&mut self, update_key : &TermReference) {
        let data_update = self.data_updates.remove(update_key).unwrap();
        self.data -= &data_update;
    }
}

impl Model {
    pub fn has_prior(&self, update_key : &TermApplication) -> bool {
        self.prior_updates.contains_key(update_key)
    }
    pub fn update_prior(&mut self, update_key : TermApplication, distr : NormalInverseWishart) {
        self.data += &distr;
        self.prior_updates.insert(update_key, distr);
    }
    pub fn downdate_prior(&mut self, key : &TermApplication) {
        let distr = self.prior_updates.remove(key).unwrap();
        self.data -= &distr;
    }
}

impl Model {
    pub fn new(feature_collections : Rc<[EnumFeatureCollection; 3]>,
              in_dimensions : usize, out_dimensions : usize) -> Model {

        let prior_updates : HashMap::<TermApplication, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<TermReference, DataUpdate> = HashMap::new();

        let mut total_feat_dims : usize = 0;
        for collection in feature_collections.iter() {
            total_feat_dims += collection.get_dimension();
        }

        let mean : Array2<f32> = Array::zeros((out_dimensions, total_feat_dims));

        let precision_mult : f32 = (1.0f32 / (PRIOR_SIGMA * PRIOR_SIGMA));
        let in_precision : Array2<f32> = precision_mult * Array::eye(total_feat_dims);
        let out_precision : Array2<f32> = precision_mult * Array::eye(out_dimensions);
        let little_v = (out_dimensions as f32) + 1.0f32;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn clone_model(model : &Model) -> Model {
        let mut result = Model::new(model.feature_collections.clone(), model.in_dimensions, model.out_dimensions);
        result.data = model.data.clone();
        result
    }
    
    fn clone_and_perturb_model(model : &Model, epsilon : f32) -> Model {
        let mut result = Model::new(model.feature_collections.clone(), model.in_dimensions, model.out_dimensions);
        result.data = model.data.clone();
        
        let mean = &model.data.mean;
        let t = mean.shape()[0];
        let s = mean.shape()[1];

        let perturbation = epsilon * random_matrix(t, s);

        result.data.mean += &perturbation;

        result.data.recompute_derived();

        result
    }

    #[test]
    fn data_updates_undo_cleanly() {
        let t = 5;
        let s = 4;
        
        let expected = random_model(s, t);

        let mut model = expected.clone();
        let data_point = random_data_point(s, t);

        model += data_point.clone();
        model -= data_point.clone();

        assert_equal_distributions_to_within(&model.data, &expected.data, 1.0f32);
    }

    #[test]
    fn sampling_accurate() {
        let epsilon = 10.0f32;
        let num_samps = 1000;
        let in_dimensions = 2;
        let out_dimensions = 2;
        let model = random_model(in_dimensions, out_dimensions);

        let model_schmear = model.get_schmear().flatten();

        let model_dims = model_schmear.mean.shape()[0];

        let mut mean = Array::zeros((model_dims,));
        let mut rng = rand::thread_rng();

        let scale_fac = 1.0f32 / (num_samps as f32);

        for i in 0..num_samps {
            let sample = model.sample_as_vec(&mut rng);

            mean += &sample;
        }

        mean *= scale_fac;

        assert_equal_vectors_to_within(&mean, &model_schmear.mean, epsilon);


        let mut covariance = Array::zeros((model_dims, model_dims));
        for i in 0..num_samps {
            let sample = model.sample_as_vec(&mut rng);

            let diff = &sample - &model_schmear.mean;
            covariance += &(scale_fac * &outer(&diff, &diff));
        }

        assert_equal_matrices_to_within(&covariance, &model_schmear.covariance, epsilon * (model_dims as f32));
    }

    #[test]
    fn find_better_app_always_better_arg_in_attraction_basin() {
        let epsilon = 0.01f32;

        let in_dimensions = 2;
        let middle_dimensions = 2;
        let out_dimensions = 2;
        let (func_model, arg_model) = random_model_app(in_dimensions, middle_dimensions, out_dimensions);

        let func_schmear = func_model.get_inverse_schmear().flatten();
        let arg_schmear = arg_model.get_inverse_schmear().flatten();

        let actual_func_mean = func_model.get_mean_as_vec();
        let actual_arg_mean = arg_model.get_mean_as_vec();

        let target = func_model.eval(&actual_arg_mean);

        let perturbed_arg_model = clone_and_perturb_model(&arg_model, epsilon);

        let perturbed_arg_mean = perturbed_arg_model.get_mean_as_vec();

        let perturbed_dist = arg_schmear.mahalanobis_dist(&perturbed_arg_mean);

        let perturbed_arg_schmear = perturbed_arg_model.get_inverse_schmear().flatten();

        let (better_func_schmear, better_arg_schmear) = 
            func_model.find_better_app(perturbed_arg_schmear, &target);

        let better_func_mean = better_func_schmear.mean;
        let better_arg_mean = better_arg_schmear.mean;

        let better_dist = arg_schmear.mahalanobis_dist(&better_arg_mean);


        //Assert that the bettered one must be closer
        if (better_dist > perturbed_dist) {
            println!("Perturbed distance {} was not bettered, became {}", perturbed_dist, better_dist);
            println!("Perturbed arg relative to basept: {}", perturbed_arg_mean - &actual_arg_mean);
            println!("Bettered arg relative to basept: {}", better_arg_mean - &actual_arg_mean);
            println!("   ");
            println!("Bettered func relative to basept: {}", better_func_mean - &actual_func_mean);
            panic!();
        }
    }

    #[test]
    fn find_better_app_no_change_unless_needed() {
        let in_dimensions = 3;
        let middle_dimensions = 4;
        let out_dimensions = 5;
        let (func_model, arg_model) = random_model_app(in_dimensions, middle_dimensions, out_dimensions);

        let actual_func_mean = func_model.get_mean_as_vec();
        let actual_arg_mean = arg_model.get_mean_as_vec();

        let target = func_model.eval(&actual_arg_mean);

        let arg_schmear = arg_model.get_inverse_schmear().flatten();

        let (better_func_schmear, better_arg_schmear) = func_model.find_better_app(arg_schmear, &target);

        let better_func_mean = better_func_schmear.mean;
        let better_arg_mean = better_arg_schmear.mean;


        assert_equal_vectors(&better_func_mean, &actual_func_mean);
        assert_equal_vectors(&better_arg_mean, &actual_arg_mean);
    }

    #[test]
    fn find_better_app_always_exact() {
        let in_dimensions = 4;
        let middle_dimensions = 5;
        let out_dimensions = 6;
        let (func_model, arg_model) = random_model_app(in_dimensions, middle_dimensions, out_dimensions);

        let arg_schmear = arg_model.get_inverse_schmear().flatten();

        let target = random_vector(out_dimensions);
        let (better_func_schmear, better_arg_schmear) = func_model.find_better_app(arg_schmear, &target);

        let better_func_mean = better_func_schmear.mean;
        let better_arg_mean = better_arg_schmear.mean;
        
        let feats = func_model.get_features(&better_arg_mean);
        let better_func_mat = better_func_mean.into_shape((out_dimensions, feats.shape()[0])).unwrap();
        let better_result = better_func_mat.dot(&feats);

        assert_equal_vectors_to_within(&better_result, &target, 0.001f32);
    }
}
