use ndarray::*;

use std::collections::HashMap;
use crate::type_id::*;
use crate::feature_space_info::*;
use crate::function_space_info::*;
use topological_sort::TopologicalSort;

///A directory of `FeatureSpaceInfo`s, indexed by [`TypeId`].
pub struct SpaceInfoDirectory {
    pub feature_spaces : Vec<FeatureSpaceInfo>
}

impl SpaceInfoDirectory {
    ///Gets the [`FeatureSpaceInfo`] for the given [`TypeId`].
    pub fn get_feature_space_info(&self, type_id : TypeId) -> &FeatureSpaceInfo {
        &self.feature_spaces[type_id]
    }
}
