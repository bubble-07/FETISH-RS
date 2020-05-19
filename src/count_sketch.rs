extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;

use rand::prelude::*;

use rand::distributions::{Bernoulli, Distribution};

pub struct CountSketch {
    in_dims : usize,
    out_dims : usize,
    pub indices : Vec<usize>,
    pub signs : Vec<f32>
}

impl CountSketch {
    pub fn new(in_dims : usize, out_dims : usize) -> CountSketch {
        //Need to initialize both indices and signs here.
        let mut indices = Vec::<usize>::with_capacity(in_dims);
        let mut signs = Vec::<f32>::with_capacity(in_dims);
        let mut rng = rand::thread_rng();
        let bernoulli_distr = Bernoulli::new(0.5).unwrap();
        for i in 0..in_dims {
            let r_one : i8 = rng.gen();
            let sign = ((r_one % 2) * 2 - 1) as f32;
            signs.push(sign);

            let r_two : usize = rng.gen(); 
            let index = r_two % out_dims;
            indices.push(index);
        }
        
        CountSketch {
            in_dims,
            out_dims,
            indices,
            signs
        }
    }
    
    pub fn get_out_dimensions(&self) -> usize {
        self.out_dims
    }

    pub fn sketch(&self, v: &Array1<f32>) -> Array1<f32> {
        let mut result = Array::zeros((self.out_dims,));
        for i in 0..self.in_dims {
            result[[i,]] += self.signs[i] * v[[i,]]; 
        }
        result
    }
}
