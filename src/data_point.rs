extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

///A weighted data-point for regressions
#[derive(Clone)]
pub struct DataPoint {
    pub in_vec: Array1<f32>,
    pub out_vec : Array1<f32>,
    pub weight : f32
}
