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
        for i in 0..dimension {
            let mut vector = Array::zeros((dimension,));
            vector[[i,]] = r32(1.0f32);
            vectors.insert(vector);
        }

        VectorSpace {
            dimension,
            vectors
        }
    }

    pub fn store_vec(&mut self, vec : Array1<R32>) {
        if (vec.shape()[0] != self.dimension) {
            panic!();
        }
        self.vectors.insert(vec);
    }
}

