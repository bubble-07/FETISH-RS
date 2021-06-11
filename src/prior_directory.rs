use crate::prior_info::*;
use std::collections::HashMap;
use crate::type_id::*;

///A directory of `PriorInfo`s, indexed by [`TypeId`].
pub struct PriorDirectory {
    pub priors : HashMap<TypeId, PriorInfo>
}

impl PriorDirectory {
    pub fn get_prior_info(&self, type_id : TypeId) -> &PriorInfo {
        self.priors.get(&type_id).unwrap()
    }
}
