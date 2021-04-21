use ndarray::*;

use std::collections::HashMap;
use crate::type_id::*;
use crate::feature_space_info::*;
use crate::function_space_info::*;
use topological_sort::TopologicalSort;

pub struct SpaceInfoDirectory {
    pub feature_spaces : Vec<FeatureSpaceInfo>
}

impl SpaceInfoDirectory {
    pub fn get_feature_space_info(&self, type_id : TypeId) -> &FeatureSpaceInfo {
        &self.feature_spaces[type_id]
    }
}
