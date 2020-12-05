extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::collections::HashMap;
use crate::type_id::*;
use crate::featurized_points_directory::*;
use crate::inverse_model::*;
use crate::params::*;
use crate::embedder_state::*;
use crate::space_info::*;
use std::rc::*;


pub struct FeaturizationInverseDirectory {
    directory : HashMap<TypeId, InverseModel>
}

impl FeaturizationInverseDirectory {
    pub fn get(&self, type_id : &TypeId) -> &InverseModel {
        self.directory.get(type_id).unwrap()
    }
    pub fn new(embedder_state : &EmbedderState) -> FeaturizationInverseDirectory {
        let mut directory = HashMap::new();
        for type_id in 0..total_num_types() {
            if (!is_vector_type(type_id)) {
                let type_space_info = embedder_state.get_space_info(&type_id);
                let in_dimensions = type_space_info.in_dimensions;
                let feature_dimensions = type_space_info.feature_dimensions;
                let feat_inv_space_info = SpaceInfo::new(feature_dimensions, in_dimensions);

                let paged_model = InverseModel::new(Rc::new(feat_inv_space_info));
                directory.insert(type_id, paged_model);
            }
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
