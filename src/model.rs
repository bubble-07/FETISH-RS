extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::space_info::*;
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
use rand::prelude::*;

use std::collections::HashMap;

#[derive(Clone)]
pub struct Model {
    pub space_info : Rc<SpaceInfo>,
    pub data : NormalInverseWishart,
    prior_updates : HashMap::<TermApplication, NormalInverseWishart>,
    pub data_updates : HashMap::<TermReference, DataUpdate>
}

pub fn to_features(feature_collections : &Vec<EnumFeatureCollection>, in_vec : &Array1<f32>) -> Array1<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_features(in_vec))
                                   .collect::<Vec<_>>();
    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();

    stack(Axis(0), &comp_views).unwrap()
}

pub fn to_jacobian(feature_collections : &Vec<EnumFeatureCollection>, in_vec : &Array1<f32>) -> Array2<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_jacobian(in_vec))
                                   .collect::<Vec<_>>();

    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();

    stack(Axis(0), &comp_views).unwrap()
}

impl Model {
    pub fn get_total_dims(&self) -> usize {
        self.data.get_total_dims()
    }
}


impl Model {
    pub fn sample(&self, rng : &mut ThreadRng) -> SampledFunction {
        let mat = self.data.sample(rng);
        SampledFunction {
            in_dimensions : self.space_info.in_dimensions,
            mat : mat,
            feature_collections : self.space_info.feature_collections.clone()
        }
    }
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        self.data.sample_as_vec(rng)
    }
    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        self.data.get_mean_as_vec()
    }

    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        self.data.get_inverse_schmear()
    }

    pub fn get_schmear(&self) -> FuncSchmear {
        self.data.get_schmear()
    }

    pub fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        to_features(&self.space_info.feature_collections, in_vec)
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
    pub fn new(space_info : Rc<SpaceInfo>) -> Model {
        let prior_updates : HashMap::<TermApplication, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<TermReference, DataUpdate> = HashMap::new();

        let mean : Array2<f32> = Array::zeros((space_info.out_dimensions, space_info.feature_dimensions));

        let precision_mult : f32 = (1.0f32 / (PRIOR_SIGMA * PRIOR_SIGMA));
        let in_precision : Array2<f32> = precision_mult * Array::eye(space_info.feature_dimensions);
        let out_precision : Array2<f32> = precision_mult * Array::eye(space_info.out_dimensions);
        let little_v = (space_info.out_dimensions as f32) + 1.0f32;

        let data = NormalInverseWishart::new(mean, in_precision, out_precision, little_v);
    
        Model {
            space_info : space_info,
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

}
