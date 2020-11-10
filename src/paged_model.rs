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
use crate::cauchy_fourier_features::*;
use crate::schmear::*;
use rand::prelude::*;

use std::ops;
use std::rc::*;

use crate::model::*;

#[derive(Clone)]
//A bunch of models for modeling branch cuts of the inverse
//of some function. Regression is performed in such a way
//that the characteristics of each of the branches is preserved,
//but at the expense of predictions being multi-valued
//and data being append-only
pub struct PagedModel {
    pub space_info : Rc<SpaceInfo>,
    pub pages : Vec<NormalInverseWishart>
}

impl ops::AddAssign<DataPoint> for PagedModel {
    fn add_assign(&mut self, other : DataPoint) {
        let feat_data = self.space_info.get_data(other);

        let num_feats = feat_data.in_vec.shape()[0];

        let feat_arg_schmear = Schmear {
            mean : feat_data.in_vec.clone(),
            covariance : Array::zeros((num_feats, num_feats))
        };
        
        let mut best_ind : usize = 0;
        let mut best_dist : f32 = f32::INFINITY;

        //Find the page whose current output on the input is closest to what we
        //were just passed -- that will be the page that we'll update
        for i in 0..self.pages.len() {
            let page = &self.pages[i];
            let page_schmear = page.get_schmear();
            let out_schmear = page_schmear.apply(&feat_arg_schmear);   
            let out_schmear_inv = out_schmear.inverse();
            let dist = out_schmear_inv.mahalanobis_dist(&feat_data.out_vec);

            if (dist < best_dist) {
                best_ind = i;
                best_dist = dist;
            }
        }

        self.pages[best_ind] += &feat_data;
    }
}

impl PagedModel {
    pub fn sample(&self, rng : &mut ThreadRng, ellipsoid : &Ellipsoid, 
                  num_function_samples : usize, num_ellipsoid_samples : usize) -> Vec<Array1<f32>> {
        let ellipsoid_sampler = EllipsoidSampler::new(ellipsoid);
        let mut feat_samples = Vec::new();
        for i in 0..num_ellipsoid_samples {
            let ellipsoid_sample = ellipsoid_sampler.sample(rng);
            let feat_sample = self.space_info.get_features(&ellipsoid_sample);
            feat_samples.push(feat_sample);
        }

        let mut result = Vec::new();

        //TODO: vectorize matrix mults in loop
        for page in self.pages.iter() {
            let page_sampler = NormalInverseWishartSampler::new(page);            
            for _ in 0..num_function_samples {
                let func_sample = page_sampler.sample(rng);
                for feat_sample in feat_samples.iter() {
                    let out_vec = func_sample.dot(feat_sample);
                    result.push(out_vec);
                }
            }
        }
        
        result
    }
}
