extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;

const LIN_REG_STRENGTH : f32 = 0.1;

pub struct LinearFeatureCollection {
    in_dimensions : usize,
    reg_strength : f32
}

impl LinearFeatureCollection {
    pub fn new(in_dimensions: usize) -> LinearFeatureCollection {
        LinearFeatureCollection {
            in_dimensions : in_dimensions,
            reg_strength : LIN_REG_STRENGTH
        }
    }
}


impl FeatureCollection for LinearFeatureCollection {
    
    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }

    fn get_dimension(&self) -> usize {
        self.in_dimensions + 1
    }

    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let single_ones = Array::ones((1,));
        stack(Axis(0), &[in_vec.view(), single_ones.view()]).unwrap()
    }

    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        let mut result : Array2<f32> = Array::zeros((self.in_dimensions + 1, self.in_dimensions));
        for i in 0..self.in_dimensions {
            result[[i, i]] = 1.0;
        }
        return result;
    }

    fn get_regularization_strength(&self) -> f32 {
        self.reg_strength
    }
}

    
