extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::data_points::*;
use crate::space_info::*;
use crate::data_update::*;
use crate::data_point::*;
use crate::pseudoinverse::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
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

use rand::prelude::*;

use std::collections::HashMap;

#[derive(Clone)]
pub struct Model {
    pub space_info : Rc<SpaceInfo>,
    pub data : NormalInverseWishart,
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

pub fn to_features_mat(feature_collections : &Vec<EnumFeatureCollection>, in_mat : &Array2<f32>) -> Array2<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_features_mat(in_mat))
                                   .collect::<Vec<_>>();
    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();
    stack(Axis(1), &comp_views).unwrap()
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
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.data.sample(rng)
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

    pub fn eval(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let feats : Array1<f32> = self.space_info.get_features(in_vec);

        self.data.eval(&feats)
    }
}

impl ops::AddAssign<DataPoint> for Model {
    fn add_assign(&mut self, other: DataPoint) {
        self.data += &self.space_info.get_data(other);
    }
}

impl ops::AddAssign<DataPoints> for Model {
    fn add_assign(&mut self, other : DataPoints) {
        self.data.update_datapoints(&self.space_info.get_data_points(other));
    }
}

impl ops::SubAssign<DataPoint> for Model {
    fn sub_assign(&mut self, other: DataPoint) {
        self.data -= &self.space_info.get_data(other);
    }
}

impl ops::AddAssign<&NormalInverseWishart> for Model {
    fn add_assign(&mut self, other : &NormalInverseWishart) {
        self.data += other;
    }
}

impl ops::SubAssign<&NormalInverseWishart> for Model {
    fn sub_assign(&mut self, other : &NormalInverseWishart) {
        self.data -= other;
    }
}

impl Model {
    pub fn new(space_info : Rc<SpaceInfo>) -> Model {
        let data = NormalInverseWishart::from_space_info(space_info.clone());
    
        Model {
            space_info : space_info,
            data : data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clone_model(model : &Model) -> Model {
        let space_info = model.space_info.clone();
        let mut result = Model::new(space_info);
        result.data = model.data.clone();
        result
    }
    
    fn clone_and_perturb_model(model : &Model, epsilon : f32) -> Model {
        let space_info = model.space_info.clone();
        let mut result = Model::new(space_info);
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
    fn data_updates_bulk_matches_incremental() {
        let t = 2;
        let s = 3;

        let mut bulk_updated = random_model(s, t);
        let mut incremental_updated = bulk_updated.clone();

        let mut data_point = random_data_point(s, t);
        data_point.weight = 1.0f32;

        let mut in_vecs = Array::zeros((1, s));
        in_vecs.row_mut(0).assign(&data_point.in_vec);
        let mut out_vecs = Array::zeros((1, t));
        out_vecs.row_mut(0).assign(&data_point.out_vec);
        let data_points = DataPoints {
            in_vecs,
            out_vecs
        };

        incremental_updated += data_point;

        bulk_updated += data_points;

        assert_equal_distributions_to_within(&incremental_updated.data, &bulk_updated.data, 1.0f32);
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
