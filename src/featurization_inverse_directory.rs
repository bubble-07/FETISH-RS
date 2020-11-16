extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::collections::HashMap;
use crate::type_id::*;
use crate::featurized_points_directory::*;
use crate::paged_model::*;
use crate::params::*;
use crate::embedder_state::*;


pub struct FeaturizationInverseDirectory {
    directory : HashMap<TypeId, PagedModel>
}

impl FeaturizationInverseDirectory {
    pub fn get(&self, type_id : &TypeId) -> &PagedModel {
        self.directory.get(type_id).unwrap()
    }
    pub fn new(embedder_state : &EmbedderState) -> FeaturizationInverseDirectory {
        let mut directory = HashMap::new();
        for type_id in 0..total_num_types() {
            let space_info = embedder_state.get_space_info(&type_id);
            let paged_model = PagedModel::new(space_info, NUM_INVERSE_PAGES);
            directory.insert(type_id, paged_model);
        }
        FeaturizationInverseDirectory {
            directory
        }
    }
}

impl ops::AddAssign<FeaturizedPointsDirectory> for FeaturizationInverseDirectory {
    fn add_assign(&mut self, mut feat_points_directory : FeaturizedPointsDirectory) {
        for (type_id, feat_points) in feat_points_directory.directory.drain() {
            let paged_model = self.directory.get_mut(&type_id).unwrap();
            paged_model.add_assign(feat_points);
        }
    }
}
