use ndarray_linalg::*;
use crate::typed_vector::*;
use ndarray::*;
use crate::type_id::*;
use crate::vector_application_result::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct ConstraintCollection {
    pub constraints : Vec<VectorApplicationResult>
}

impl ConstraintCollection {
    pub fn update_repeat(&mut self, repeats : usize) {
        let mut result = Vec::new();
        for _ in 0..repeats {
            for constraint in &self.constraints {
                result.push(constraint.clone());
            }
        }
        self.constraints = result;
    }
    pub fn update_shuffle(&mut self) {
        let mut rng = thread_rng();
        self.constraints.shuffle(&mut rng);
    }
}
