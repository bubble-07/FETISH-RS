extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::sized_determinant::*;

#[derive(Clone)]
pub struct DetWeightedPoint {
    pub det : SizedDeterminant,
    pub vec : Array1<f32>
}

impl DetWeightedPoint {
    pub fn from_vector(vec : Array1<f32>) -> DetWeightedPoint {
        let size = vec.shape()[0];
        let det = SizedDeterminant::eye(size);
        DetWeightedPoint {
            det,
            vec
        }
    }
}
