extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use std::collections::HashSet;
use noisy_float::prelude::*;
use crate::array_utils::*;
use crate::inverse_schmear::*;
use crate::sampled_function::*;

pub struct VectorSpace {
    dimension : usize,
    vectors : HashSet<Array1<R32>>
}

impl VectorSpace {
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn new(dimension : usize) -> VectorSpace {
        let mut vectors : HashSet<Array1<R32>> = HashSet::new();
        vectors.insert(Array::zeros((dimension,)));
        VectorSpace {
            dimension,
            vectors
        }
    }

    pub fn store_vec(&mut self, vec : Array1<R32>) {
        self.vectors.insert(vec);
    }

    pub fn get_best_vector_arg(&self, sampled_function : &SampledFunction, target : &InverseSchmear) -> 
                          (Array1<f32>, f32) {
        let mut result_vec : Array1<f32> = Array::zeros((self.dimension,));
        let mut result_dist : f32 = f32::INFINITY;
        for vector in self.vectors.iter() {
            let as_floating = from_noisy(vector);
            let temp = sampled_function.apply(&as_floating);
            let temp_dist = target.mahalanobis_dist(&temp);

            if (temp_dist < result_dist) {
                result_vec = temp;
                result_dist = temp_dist;
            }
        }
        (result_vec, result_dist)
    }
}
