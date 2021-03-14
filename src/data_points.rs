extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

//Data points as input and output
//matrices, where each row is another data-point

pub struct DataPoints {
    pub in_vecs : Array2<f32>,
    pub out_vecs : Array2<f32>
}

impl DataPoints {
    pub fn num_points(&self) -> usize {
        self.in_vecs.shape()[0]
    }
}
