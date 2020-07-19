extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;

use crate::inverse_schmear::*;
use crate::cauchy_fourier_features::*;
use crate::func_scatter_tensor::*;

pub struct FuncInverseSchmear {
    pub mean : Array2<f32>,
    pub precision : FuncScatterTensor
}
impl FuncInverseSchmear {
    pub fn flatten(&self) -> InverseSchmear {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let mean = self.mean.clone().into_shape((t * s,)).unwrap();
        let precision = self.precision.flatten();
        InverseSchmear {
            mean,
            precision
        }
    }
}
