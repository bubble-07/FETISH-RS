extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::data_point::*;
use crate::schmear::*;
use crate::sigma_points::*;
use crate::model::*;

#[derive(Clone)]
pub struct DataUpdate { 
    pub in_vec : Array1<f32>,
    pub out_sigma_points : Vec<Array1<f32>>
}

impl DataUpdate {
    pub fn featurize(&self, model : &Model) -> DataUpdate {
        let feat_vec = model.get_features(&self.in_vec);
        let out_sigma_points = self.out_sigma_points.clone();
        let result = DataUpdate {
            in_vec : feat_vec,
            out_sigma_points
        };
        result
    }
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

pub fn construct_vector_data_update(in_vec : Array1<f32>, out_vec : Array1<f32>) -> DataUpdate {
    let mut out_sigma_points = Vec::new();
    out_sigma_points.push(out_vec);

    DataUpdate {
        in_vec,
        out_sigma_points
    }
}

pub fn construct_data_update(in_vec : Array1<f32>, out_schmear : &Schmear) -> DataUpdate {
    let out_sigma_points = get_sigma_points(out_schmear);
    DataUpdate {
        in_vec,
        out_sigma_points
    }
}
