extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::ops;
use std::rc::*;

use crate::feature_space_info::*;
use crate::data_points::*;
use crate::sigma_points::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::alpha_formulas::*;
use crate::vector_space::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::data_point::*;
use crate::function_space_info::*;
use rand::prelude::*;
use crate::type_id::*;

extern crate pretty_env_logger;

use std::collections::HashMap;

pub struct FunctionSpaceDirectory {
    pub directory : HashMap<TypeId, FunctionSpaceInfo>
}

impl FunctionSpaceDirectory {
    pub fn get_feature_space_info(&self, type_id : TypeId) -> &Rc<FeatureSpaceInfo> {
        &self.directory.get(&type_id).unwrap().func_feat_info
    }
}
