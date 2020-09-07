extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::pseudoinverse::*;
use crate::func_scatter_tensor::*;
use crate::func_inverse_schmear::*;
use crate::func_schmear::*;
use crate::schmear::*;
use crate::data_point::*;
use crate::wishart::*;
use crate::sherman_morrison::*;
use crate::inverse_schmear::*;
use crate::cauchy_fourier_features::*;
use crate::linalg_utils::*;
use crate::sized_determinant::*;
use crate::det_weighted_point::*;
use crate::normal_inverse_wishart_sampler::*;

use rand::prelude::*;
use rand_distr::{Cauchy, Distribution};
use rand_distr::StandardNormal;


///Normal-inverse-wishart distribution representation
///for bayesian inference
pub struct NormalInverseWishart {
    pub mean : Array2<f32>,
    pub precision_u : Array2<f32>,
    pub precision : Array2<f32>,
    pub sigma : Array2<f32>,
    pub big_v : Array2<f32>,
    pub little_v : f32,
    pub t : usize,
    pub s : usize
}

pub fn mean_to_array(mean : &Array2<f32>) -> Array1<f32> {
    let t = mean.shape()[0];
    let s = mean.shape()[1];
    let n = t * s;

    let mut mean_copy = Array::zeros((t, s));
    mean_copy.assign(mean);

    mean_copy.into_shape((n,)).unwrap()
}

impl NormalInverseWishart {
    pub fn get_total_dims(&self) -> usize {
        self.s * self.t
    }

    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let sampler = NormalInverseWishartSampler::new(&self);
        sampler.sample(rng)
    }

    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1<f32> {
        let thick = self.sample(rng);
        let total_dims = self.get_total_dims();
        thick.into_shape((total_dims,)).unwrap()
    }

    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        mean_to_array(&self.mean)
    }
    pub fn get_mean(&self) -> Array2<f32> {
        self.mean.clone()
    }
    pub fn get_det_weighted_mean(&self) -> DetWeightedPoint {
        let scale = self.little_v - (self.t as f32) - 1.0f32;
        let in_det = SizedDeterminant::from_psd_matrix(&self.precision);
        let mut out_det = SizedDeterminant::from_psd_matrix(&self.big_v);
        out_det.invert();
        out_det.scale(scale);
        let total_det = in_det.tensor(&out_det);

        let mean = self.get_mean_as_vec();
        DetWeightedPoint {
            det : total_det,
            vec : mean
        }
    }

    pub fn get_schmear(&self) -> FuncSchmear {
        FuncSchmear {
            mean : self.mean.clone(),
            covariance : self.get_covariance()
        }
    }

    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        FuncInverseSchmear {
            mean : self.mean.clone(),
            precision : self.get_precision()
        }
    }

    pub fn get_precision(&self) -> FuncScatterTensor {
        let scale = self.little_v - (self.t as f32) - 1.0f32;
        let mut out_precision = pseudoinverse_h(&self.big_v);
        out_precision *= scale;
        FuncScatterTensor::from_in_and_out_scatter(self.precision.clone(), out_precision)
    }

    pub fn get_covariance(&self) -> FuncScatterTensor {
        let scale = 1.0f32 / (self.little_v - (self.t as f32) - 1.0f32);
        let big_v_scaled = scale * &self.big_v;
        FuncScatterTensor::from_in_and_out_scatter(self.sigma.clone(), big_v_scaled)
    }

    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        self.mean.dot(in_vec)
    }
}

impl NormalInverseWishart {
    pub fn new(mean : Array2<f32>, precision : Array2<f32>, big_v : Array2<f32>, little_v : f32) -> NormalInverseWishart {
        let precision_u : Array2<f32> = mean.dot(&precision);
        let sigma : Array2<f32> = precision.invh().unwrap();
        let t = mean.shape()[0];
        let s = mean.shape()[1];

        NormalInverseWishart {
            mean,
            precision_u,
            precision,
            sigma,
            big_v,
            little_v,
            t,
            s
        }
    }
}

///Allows doing dist ^= dist to invert dist in place
impl ops::BitXorAssign<()> for NormalInverseWishart {
    fn bitxor_assign(&mut self, rhs: ()) {
        self.precision_u *= -1.0;
        self.precision *= -1.0;
        self.sigma *= -1.0;
        self.little_v = 2.0 * (self.t as f32) - self.little_v;
        self.big_v *= -1.0;
    }
}

fn zero_normal_inverse_wishart(t : usize, s : usize) -> NormalInverseWishart {
    NormalInverseWishart {
        mean: Array::zeros((t, s)),
        precision_u: Array::zeros((t, s)),
        precision: Array::zeros((t, s)),
        sigma: Array::zeros((t, s)),
        big_v: Array::zeros((t, s)),
        little_v: (t as f32),
        t,
        s
    }
}

impl NormalInverseWishart {
    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let x = &data_point.in_vec;
        let y = &data_point.out_vec;
        let s = if (downdate) {-1.0f32} else {1.0f32};
        let w = data_point.weight * s;

        let mut out_precision = self.precision.clone();
        sherman_morrison_update(&mut out_precision, &mut self.sigma, w, x);

        self.precision_u += &(w * outer(y, x));

        let out_mean = self.precision_u.dot(&self.sigma);

        self.little_v += s;

        let mean_diff = &out_mean - &self.mean;
        let r = y - &out_mean.dot(x);

        self.big_v += &outer(&r, &r);
        self.big_v += &mean_diff.dot(&self.precision).dot(&mean_diff.t());

        self.mean = out_mean;
        self.precision = out_precision;
    }
}

impl ops::AddAssign<&DataPoint> for NormalInverseWishart {
    fn add_assign(&mut self, other: &DataPoint) {
        self.update(other, false)
    }
}

impl ops::SubAssign<&DataPoint> for NormalInverseWishart {
    fn sub_assign(&mut self, other: &DataPoint) {
        self.update(other, true)
    }
}

impl NormalInverseWishart {
    fn update_combine(&mut self, other : &NormalInverseWishart, downdate : bool) {
        let s = if (downdate) {-1.0f32} else {1.0f32};
        
        let mut other_precision = other.precision.clone();
        other_precision *= s;

        let mut other_big_v = other.big_v.clone();
        other_big_v *= s;

        let other_mean = &other.mean;
        let other_little_v = if (downdate) {(self.t as f32) * 2.0f32 - other.little_v} else {other.little_v};

        let mut other_precision_u = other.precision_u.clone();
        other_precision_u *= s;


        self.precision_u += &other_precision_u;

        let precision_out = &self.precision + &other_precision;

        self.sigma = pseudoinverse_h(&precision_out);

        let mean_out = self.precision_u.dot(&self.sigma);

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = other_mean - &mean_out;
        
        let u_diff_l_u_diff_one = mean_one_diff.dot(&self.precision).dot(&mean_one_diff.t());
        let u_diff_l_u_diff_two = mean_two_diff.dot(&other_precision).dot(&mean_two_diff.t());

        self.little_v += other_little_v - (self.t as f32);
        self.precision = precision_out;
        self.mean = mean_out;

        self.big_v += &other_big_v;
        self.big_v += &u_diff_l_u_diff_one;
        self.big_v += &u_diff_l_u_diff_two;
    }
}

impl ops::AddAssign<&NormalInverseWishart> for NormalInverseWishart {
    fn add_assign(&mut self, other: &NormalInverseWishart) {
        self.update_combine(other, false);
    }
}
impl ops::SubAssign<&NormalInverseWishart> for NormalInverseWishart {
    fn sub_assign(&mut self, other : &NormalInverseWishart) {
        self.update_combine(other, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_convergence_noiseless() {
        let num_samps = 1000;
        let s = 5;
        let t = 4;
        let mut model = standard_normal_inverse_gamma(s, t);

        let mat = random_matrix(t, s);
        for i in 0..num_samps {
            let vec = random_vector(s);
            let out = mat.dot(&vec);
            let out_precision = 100.0f32 * random_psd_matrix(t);

            let out_inv_schmear = InverseSchmear {
                mean : out,
                precision : out_precision
            };

            let data_point = DataPoint {
                in_vec : vec,
                out_inv_schmear
            };

            model += &data_point;
        }

        assert_equal_matrices(&model.mean, &mat);
    }
}
