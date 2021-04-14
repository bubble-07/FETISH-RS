extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::data_point::*;
use crate::schmear::*;
use crate::sigma_points::*;
use crate::function_space_info::*;

///Economical representation of a collection of [`DataPoint`]s,
///all with the same input vector, but differing output vectors,
///whose weights are uniform and sum to one.
#[derive(Clone)]
pub struct DataUpdate { 
    pub in_vec : Array1<f32>,
    pub out_sigma_points : Vec<Array1<f32>>
}

impl DataUpdate {
    ///Assuming that this [`DataUpdate`] is given in terms of the base space
    ///of the input space of a given model and its output space, and 
    ///given the [`FunctionSpaceInfo`] of the model that this [`DataUpdate`] 
    ///will be used to update, computes a new [`DataUpdate`] whose input
    ///has been passed through the [`FunctionSpaceInfo`]'s input feature mapping.
    pub fn featurize(&self, func_space_info : &FunctionSpaceInfo) -> DataUpdate {
        let feat_vec = func_space_info.in_feat_info.get_features(self.in_vec.view());
        let out_sigma_points = self.out_sigma_points.clone();
        let result = DataUpdate {
            in_vec : feat_vec,
            out_sigma_points
        };
        result
    }
    ///Gets the vector of [`DataPoint`]s which this [`DataUpdate`] represents.
    pub fn get_data_points(&self) -> Vec<DataPoint> {
        let mut result = Vec::new(); 

        let n = self.out_sigma_points.len();
        let weight = 1.0f32 / (n as f32);
        for sigma_point in &self.out_sigma_points {
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

///Constructs a [`DataUpdate`] corresponding to a single input/output pair.
pub fn construct_vector_data_update(in_vec : Array1<f32>, out_vec : Array1<f32>) -> DataUpdate {
    let mut out_sigma_points = Vec::new();
    out_sigma_points.push(out_vec);

    DataUpdate {
        in_vec,
        out_sigma_points
    }
}

///Constructs a [`DataUpdate`] with the given input and outputs constructed from the
///[`get_sigma_points`] of the passed output [`Schmear`]
pub fn construct_data_update(in_vec : Array1<f32>, out_schmear : &Schmear) -> DataUpdate {
    let out_sigma_points = get_sigma_points(out_schmear);
    DataUpdate {
        in_vec,
        out_sigma_points
    }
}
