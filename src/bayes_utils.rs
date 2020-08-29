extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::linalg_utils::*;
use crate::test_utils::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::func_scatter_tensor::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::cauchy_fourier_features::*;

use rand::prelude::*;
use rand_distr::{Cauchy, Distribution};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;

///Data point [input, output pair]
///with an output precision matrix
pub struct DataPoint {
    pub in_vec: Array1<f32>,
    pub out_inv_schmear : InverseSchmear
}

///Normal-inverse-gamma distribution representation
///for bayesian inference
#[derive(Clone)]
pub struct NormalInverseGamma {
    pub mean: Array2<f32>,
    precision_u: Array2<f32>,
    precision : FuncScatterTensor,
    sigma : FuncScatterTensor,
    a: f32,
    b: f32,
    t: usize,
    s: usize
}

pub fn mean_to_array(mean : &Array2<f32>) -> Array1<f32> {
    let t = mean.shape()[0];
    let s = mean.shape()[1];
    let n = t * s;

    let mut mean_copy = Array::zeros((t, s));
    mean_copy.assign(mean);

    mean_copy.into_shape((n,)).unwrap()
}


impl NormalInverseGamma {

    pub fn get_total_dims(&self) -> usize {
        self.s * self.t
    }

    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let sample = self.sample(rng);

        let result = sample.into_shape((t * s,)).unwrap();

        result
    }

    pub fn sample(&self, rng : &mut ThreadRng) -> Array2::<f32> {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];
        let std_norm_samp = Array::random((t, s), StandardNormal);

        let covariance = self.get_covariance();
        let sqrt_covariance = covariance.sqrt();
        let mut result : Array2<f32> = sqrt_covariance.transform(&std_norm_samp);
       
        //Add the mean to offset it right
        result += &self.mean;
        result
    }

    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        mean_to_array(&self.mean)
    }
    pub fn get_mean(&self) -> Array2::<f32> {
        self.mean.clone()
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
        let mut result = self.precision.clone();
        result *= ((self.a - 1.0f32) / self.b);
        result
    }
    pub fn get_covariance(&self) -> FuncScatterTensor {
        let mut result = self.sigma.clone();
        result *= (self.b / (self.a - 1.0f32));
        result
    }
}

impl NormalInverseGamma {
    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        self.mean.dot(in_vec)
    }
}

impl NormalInverseGamma {
    pub fn recompute_derived(&mut self) {
        self.precision_u = self.precision.transform(&self.mean);
        self.sigma = self.precision.inverse();
    }
    pub fn new(mean : Array2<f32>, precision : FuncScatterTensor, a : f32, b : f32, t : usize, s : usize) -> NormalInverseGamma {
        let precision_u = precision.transform(&mean);
        let sigma = precision.inverse();

        NormalInverseGamma {
            mean,
            precision_u,
            precision,
            sigma,
            a,
            b,
            t,
            s
        }
    }
}

impl NormalInverseGamma {

    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let mut out_precision = data_point.out_inv_schmear.precision.clone();
        if (downdate) {
            out_precision *= -1.0f32;
        }

        let in_precision = outer(&data_point.in_vec, &data_point.in_vec);

        let precision_contrib = FuncScatterTensor::from_in_and_out_scatter(in_precision, out_precision.clone());

        self.precision += &precision_contrib;

        self.sigma = self.precision.inverse();

        let data_out_mean : &Array1::<f32> = &data_point.out_inv_schmear.mean;

        let out_precision_y : Array1<f32> = out_precision.dot(data_out_mean);

        let x_out_precision_y = outer(&out_precision_y, &data_point.in_vec);

        let y_T_out_precision_y = out_precision_y.dot(data_out_mean);
        
        let u_precision_u_zero = frob_inner(&self.mean, &self.precision_u);

        self.b += 0.5 * y_T_out_precision_y;
        self.b += 0.5 * u_precision_u_zero;

        self.precision_u += &x_out_precision_y;
        self.mean = self.sigma.transform(&self.precision_u);

        let u_precision_u_n = frob_inner(&self.mean, &self.precision_u);
        self.b -= 0.5 * &u_precision_u_n;

        if (self.b < 0.0f32) {
            println!("self.b is {}", self.b);
            panic!();
        }

        self.a += (if downdate == true {-0.5} else {0.5});
        if (self.a < 0.0f32) {
            panic!();
        }
    }
}

impl ops::AddAssign<&DataPoint> for NormalInverseGamma {
    fn add_assign(&mut self, other: &DataPoint) {
        self.update(other, false)
    }
}

impl ops::SubAssign<&DataPoint> for NormalInverseGamma {
    fn sub_assign(&mut self, other: &DataPoint) {
        self.update(other, true)
    }
}

impl NormalInverseGamma {
    fn update_combine(&mut self, other : &NormalInverseGamma, downdate : bool) {
        let s = if (downdate) {-1.0f32} else {1.0f32};

        let mut other_precision = other.precision.clone();
        other_precision *= s;

        let other_mean = &other.mean;
        let other_precision_u = s * other.precision_u.clone();
        let other_a = if (downdate) { -(self.s as f32) - other.a } else { other.a };
        let other_b = s * other.b;

        let mut precision_out = self.precision.clone();
        precision_out += &other_precision;

        let precision_u_out = &self.precision_u + &other.precision_u;

        let sigma_out = precision_out.inverse();

        let mean_out = sigma_out.transform(&precision_u_out);

        let a_out = self.a + other.a + 0.5f32 * (self.s as f32);

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = other_mean - &mean_out;

        let u_diff_l_u_diff_one = self.precision.inner_product(&mean_one_diff, &mean_one_diff);
        let u_diff_l_u_diff_two = other.precision.inner_product(&mean_two_diff, &mean_two_diff);

        let b_out = self.b + other.b + 0.5f32 * (u_diff_l_u_diff_one + u_diff_l_u_diff_two);


        if (b_out < 0.0f32) {
            println!("U diff l u diff one {}", u_diff_l_u_diff_one);
            println!("U diff l u diff two {}", u_diff_l_u_diff_two);
            println!("self b {}", self.b);
            println!("other b {}", other.b);
            panic!();
        }

        if (a_out < 0.0f32) {
            panic!();
        }

        self.mean = mean_out;
        self.precision_u = precision_u_out;
        self.precision = precision_out;
        self.sigma = sigma_out;

        self.a = a_out;
        self.b = b_out;
    }
}

impl ops::AddAssign<&NormalInverseGamma> for NormalInverseGamma {
    fn add_assign(&mut self, other: &NormalInverseGamma) {
        self.update_combine(other, false);
    }
}
impl ops::SubAssign<&NormalInverseGamma> for NormalInverseGamma {
    fn sub_assign(&mut self, other : &NormalInverseGamma) {
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
