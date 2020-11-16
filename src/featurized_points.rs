extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;

use std::rc::*;

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
}
