extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

///Data point [input, output pair]
///with an output precision matrix
#[derive(Clone)]
pub struct DataPoint {
    pub in_vec: Array1<f32>,
    pub out_vec : Array1<f32>,
    pub weight : f32
}
