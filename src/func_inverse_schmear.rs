extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::func_schmear::*;

use crate::inverse_schmear::*;
use crate::func_scatter_tensor::*;

#[derive(Clone)]
pub struct FuncInverseSchmear {
    pub mean : Array2<f32>,
    pub precision : FuncScatterTensor
}
impl FuncInverseSchmear {
    pub fn inverse(&self) -> FuncSchmear {
        let mean = self.mean.clone();
        let covariance = self.precision.inverse();
        FuncSchmear {
            mean,
            covariance
        }
    }
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
