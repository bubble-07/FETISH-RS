extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::space_info::*;
use crate::data_point::*;
use crate::ellipsoid::*;
use crate::ellipsoid_sampler::*;
use crate::normal_inverse_wishart_sampler::*;
use crate::normal_inverse_wishart::*;
use crate::rand_utils::*;
use crate::featurized_points::*;
use crate::schmear::*;
use rand::prelude::*;

use std::ops;
use std::rc::*;

use crate::model::*;

#[derive(Clone)]
//A model for the inverse of the featurization map.
pub struct InverseModel {
    pub model : Model
}

impl InverseModel {
    pub fn new(space_info : Rc<SpaceInfo>) -> InverseModel {
        let mut model = Model::new(space_info);
        //Adjust to ensure that sampling from it is well-defined,
        //even from the outset
        model.data.little_v += 2.0f32;
        InverseModel {
            model
        }
    }
}

impl ops::AddAssign<FeaturizedPoints> for InverseModel {
    fn add_assign(&mut self, feat_points : FeaturizedPoints) {
        let data_points = feat_points.to_feat_inverse_data_points();
        self.model.add_assign(data_points);
    }
}

impl InverseModel {
    pub fn sample_single_inverse_point(&self, rng : &mut ThreadRng, in_vec : &Array1<f32>) -> Array1<f32> {
        let model_sampler = NormalInverseWishartSampler::new(&self.model.data);
        let func_sample = model_sampler.sample(rng);

        let feat_vec = self.model.space_info.get_features(in_vec);

        let result = func_sample.dot(&feat_vec);
        result
    }

    pub fn sample_ellipsoid_inverse(&self, rng : &mut ThreadRng, ellipsoid : &Ellipsoid, 
                  num_function_samples : usize, num_ellipsoid_samples : usize) -> Vec<Array1<f32>> {
        let ellipsoid_sampler = EllipsoidSampler::new(ellipsoid);
        let mut feat_samples = Vec::new();
        for i in 0..num_ellipsoid_samples {
            let ellipsoid_sample = ellipsoid_sampler.sample(rng);
            let feat_sample = self.model.space_info.get_features(&ellipsoid_sample);
            feat_samples.push(feat_sample);
        }

        let mut result = Vec::new();

        //TODO: vectorize matrix mults in loop
        let model_sampler = NormalInverseWishartSampler::new(&self.model.data);            
        for _ in 0..num_function_samples {
            let func_sample = model_sampler.sample(rng);
            for feat_sample in feat_samples.iter() {
                let out_vec = func_sample.dot(feat_sample);
                result.push(out_vec);
            }
        }
    
        result

    }
}
