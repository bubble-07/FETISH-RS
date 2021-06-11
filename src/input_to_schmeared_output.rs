extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::data_point::*;
use crate::schmear::*;
use crate::sigma_points::*;
use crate::function_space_info::*;

use serde::{Serialize, Deserialize};

///A variation on [`DataPoint`] where the output
///is allowed to be a schmear instead of just one point,
///and the weight is assumed to be one.
#[derive(Clone, Serialize, Deserialize)]
pub struct InputToSchmearedOutput {
    pub in_vec : Array1<f32>,
    pub out_schmear : Schmear
}

impl InputToSchmearedOutput {
    ///Assuming that this [`InputToSchmearedOutput`] is given in terms of the base space
    ///of the input space of a given model and its output space, and 
    ///given the [`FunctionSpaceInfo`] of the model that this [`InputToSchmearedOutput`] 
    ///will be used to update, computes a new [`InputToSchmearedOutput`] whose input
    ///has been passed through the [`FunctionSpaceInfo`]'s input feature mapping.
    pub fn featurize(&self, func_space_info : &FunctionSpaceInfo) -> InputToSchmearedOutput {
        let feat_vec = func_space_info.in_feat_info.get_features(self.in_vec.view());
        let out_schmear = self.out_schmear.clone();
        let result = InputToSchmearedOutput {
            in_vec : feat_vec,
            out_schmear
        };
        result
    }
    ///Gets the vector of [`DataPoint`]s which this [`InputToSchmearedOutput`] represents.
    pub fn get_data_points(&self) -> Vec<DataPoint> {
        let mut result = Vec::new(); 

        let out_sigma_points = get_sigma_points(&self.out_schmear);

        let n = out_sigma_points.len();
        let weight = 1.0f32 / (n as f32);
        for sigma_point in &out_sigma_points {
            let data_point = DataPoint {
                in_vec : self.in_vec.clone(),
                out_vec : sigma_point.clone(),
                weight
            };
            result.push(data_point);
        }
        result
    }
}
