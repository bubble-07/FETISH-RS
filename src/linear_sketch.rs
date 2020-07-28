extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use crate::schmear::*;

use crate::inverse_schmear::*;
use crate::params::*;
use crate::pseudoinverse::*;

pub struct LinearSketch {
    projection_mat : Array2<f32>,
    projection_mat_pinv : Array2<f32>
}

impl LinearSketch {
    pub fn new(in_dimensions : usize, out_dimensions : usize) -> LinearSketch {
        let projection_mat = Array::random((out_dimensions, in_dimensions), StandardNormal);
        let projection_mat_pinv = pseudoinverse(&projection_mat);
        LinearSketch {
            projection_mat,
            projection_mat_pinv
        }
    }
    pub fn compress_inverse_schmear(&self, inv_schmear : &InverseSchmear) -> InverseSchmear {
        inv_schmear.transform_compress(&self.projection_mat, &self.projection_mat_pinv)
    }
    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        schmear.transform_compress(&self.projection_mat)
    }

    pub fn sketch(&self, vec : &Array1<f32>) -> Array1<f32> {
        self.projection_mat.dot(vec)
    }
    pub fn expand(&self, mean : &Array1<f32>) -> Array1<f32> {
        self.projection_mat_pinv.dot(mean)
    }
    pub fn get_output_dimension(&self) -> usize {
        self.projection_mat.shape()[0]
    }
    pub fn get_input_dimension(&self) -> usize {
        self.projection_mat.shape()[1]
    }
}
