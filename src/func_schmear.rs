extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;

use crate::inverse_schmear::*;
use crate::schmear::*;
use crate::linalg_utils::*;
use crate::cauchy_fourier_features::*;
use crate::func_scatter_tensor::*;

pub struct FuncSchmear {
    pub mean : Array2<f32>,
    pub covariance : FuncScatterTensor
}

impl FuncSchmear {
    pub fn flatten(&self) -> Schmear {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let mean = self.mean.clone().into_shape((t * s,)).unwrap();
        let covariance = self.covariance.flatten();
        Schmear {
            mean,
            covariance
        }
    }
    ///Computes the output schmear of this func schmear applied
    ///to a given argument schmear
    pub fn apply(&self, x : &Schmear) -> Schmear {
        let sigma_dot_u = frob_inner(&self.covariance.in_scatter, &x.covariance);
        let u_inner_product = self.covariance.in_scatter.dot(&x.mean).dot(&x.mean);
        let v_scale = (sigma_dot_u + u_inner_product) * self.covariance.scale;
        let v_contrib = v_scale * &self.covariance.out_scatter;

        let m_sigma_m_t = self.mean.dot(&x.covariance).dot(&self.mean.t());

        let result_covariance = v_contrib + &m_sigma_m_t;
        let result_mean = self.mean.dot(&x.mean);
        let result = Schmear {
            mean : result_mean,
            covariance : result_covariance
        };
        result
    }
}
