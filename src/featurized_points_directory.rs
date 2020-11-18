extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use std::ops;

use std::collections::HashMap;
use crate::featurized_points::*;
use crate::type_id::*;
use crate::sampled_embedder_state::*;

pub struct FeaturizedPointsDirectory {
    pub directory : HashMap<TypeId, FeaturizedPoints>
}

impl FeaturizedPointsDirectory {
    pub fn new(embedder_state : &SampledEmbedderState) -> FeaturizedPointsDirectory {
        let mut directory = HashMap::new();
        for type_id in 0..total_num_types() {
            let space_info = embedder_state.get_space_info(&type_id);
            let feat_points = FeaturizedPoints::new(space_info);
            directory.insert(type_id, feat_points);
        }
        FeaturizedPointsDirectory {
            directory
        }
    }
    pub fn get_space(&mut self, type_id : &TypeId) -> &mut FeaturizedPoints {
        self.directory.get_mut(type_id).unwrap()
    }
}

impl ops::AddAssign<FeaturizedPointsDirectory> for FeaturizedPointsDirectory {
    fn add_assign(&mut self, mut other : FeaturizedPointsDirectory) {
        for (type_id, other_feat_points) in other.directory.drain() {
            let my_feat_points = self.directory.get_mut(&type_id).unwrap();
            my_feat_points.add_assign(other_feat_points);
        }
    }
}
