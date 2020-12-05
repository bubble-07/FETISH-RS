extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

use std::rc::*;
use std::ops;
use crate::data_points::*;

use crate::space_info::*;

use std::collections::HashMap;

pub struct FeaturizedPoints {
    space_info : Rc<SpaceInfo>,
    pub points : Vec<(Array1<f32>, Array1<f32>)>
}

impl FeaturizedPoints {
    pub fn new(space_info : Rc<SpaceInfo>) -> FeaturizedPoints {
        FeaturizedPoints {
            space_info : space_info,
            points : Vec::new()
        }
    }
    pub fn get_space_info(&self) -> Rc<SpaceInfo> {
        self.space_info.clone()
    }
    pub fn get_features(&mut self, in_vec : &Array1<f32>) -> &Array1<f32> {
        let feat_vec = self.space_info.get_features(in_vec);
        let pair = (in_vec.clone(), feat_vec);
        self.points.push(pair);
        &self.points[self.points.len() - 1].1
    }

    pub fn to_feat_inverse_data_points(&self) -> DataPoints {
        let num_points = self.points.len();
        let num_feats = self.space_info.feature_dimensions;
        let num_in_dims = self.space_info.in_dimensions;

        let mut feat_mat = Array::zeros((num_points, num_feats));
        let mut in_vec_mat = Array::zeros((num_points, num_in_dims));
        let mut i : usize = 0;
        for (in_vec, feat_vec) in &self.points {
            feat_mat.row_mut(i).assign(feat_vec);
            in_vec_mat.row_mut(i).assign(in_vec);
            i += 1; 
        }
        DataPoints {
            in_vecs : feat_mat,
            out_vecs : in_vec_mat
        }
    }
}

impl ops::AddAssign<FeaturizedPoints> for FeaturizedPoints {
    fn add_assign(&mut self, mut other : FeaturizedPoints) {
        self.points.append(&mut other.points);
    }
}
