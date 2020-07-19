extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::linalg_utils::*;
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
pub struct NormalInverseGamma {
    mean: Array2<f32>,
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

fn tensors_to_schmeary(mean : &Array2<f32>, sigma : &Array4<f32>) -> (Array1<f32>, Array2<f32>) {
    let t = mean.shape()[0];
    let s = mean.shape()[1];
    let n = t * s;

    let mut sigma_copy = Array::zeros((t, s, t, s));

    sigma_copy.assign(sigma);
    
    let flat_sigma = sigma_copy.into_shape((n, n)).unwrap();
    let flat_mean : Array1<f32> = mean_to_array(mean);

    (flat_mean, flat_sigma)
}

pub fn tensors_to_schmear(mean : &Array2<f32>, sigma : &Array4<f32>) -> Schmear {
    let (mean, covariance) = tensors_to_schmeary(mean, sigma);
    Schmear {
        mean,
        covariance
    }
}

pub fn tensors_to_inv_schmear(mean : &Array2<f32>, precision : &Array4<f32>) -> InverseSchmear {
    let (mean, precision) = tensors_to_schmeary(mean, precision);
    InverseSchmear {
        mean,
        precision
    }
}

pub fn schmear_to_tensors(t : usize, s : usize, schmear : &Schmear) -> (Array2<f32>, Array4<f32>) {
    let n = t * s; 
    
    let mut mean_copy = Array::zeros((n,));
    let mut sigma_copy = Array::zeros((n, n));
    
    mean_copy.assign(&schmear.mean);
    sigma_copy.assign(&schmear.covariance);

    let inflate_mean = mean_copy.into_shape((t, s)).unwrap();
    let inflate_sigma = sigma_copy.into_shape((t, s, t, s)).unwrap();
    (inflate_mean, inflate_sigma)
}

impl NormalInverseGamma {

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
        let sqrt_covariance = self.sigma.sqrt();
        let mut result : Array2<f32> = sqrt_covariance.transform(&std_norm_samp);
       
        //Great, now we need to sample from the inverse-gamma part
        //to determine a multiplier for the covariance
        let inv_gamma_sample = gen_inverse_gamma_random(rng, self.a, self.b);
        result *= inv_gamma_sample;

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
        result *= (self.b / self.a);
        result
    }
    pub fn get_covariance(&self) -> FuncScatterTensor {
        let mut result = self.sigma.clone();
        result *= (self.a / self.b);
        result
    }
}

impl NormalInverseGamma {
    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        self.mean.dot(in_vec)
    }
}

impl NormalInverseGamma {
    pub fn new(mean : Array2<f32>, precision : FuncScatterTensor, a : f32, b : f32, t : usize, s : usize) -> NormalInverseGamma {
        println!("Transforming mean with dims {}, {}", t, s);
        let precision_u = precision.transform(&mean);
        println!("Inverting precision with dims {}, {}", t, s);
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
        let out_precision = &data_point.out_inv_schmear.precision;

        let in_precision = outer(&data_point.in_vec, &data_point.in_vec);

        let precision_contrib = FuncScatterTensor::from_in_and_out_scatter(in_precision, out_precision.clone());

        if (downdate) {
            self.precision -= &precision_contrib;
        } else {
            self.precision += &precision_contrib;
        }


        self.sigma = self.precision.inverse();

        let data_out_mean : &Array1::<f32> = &data_point.out_inv_schmear.mean;

        let out_precision_y : Array1<f32> = out_precision.dot(data_out_mean);

        let mut x_out_precision_y = outer(&out_precision_y, &data_point.in_vec);

        let mut y_T_out_precision_y = out_precision_y.dot(data_out_mean);

        if (downdate == true) {
            x_out_precision_y *= -1.0f32;
            y_T_out_precision_y *= -1.0f32;
        }
        
        let u_precision_u_zero = frob_inner(&self.mean, &self.precision_u);

        self.b += 0.5 * y_T_out_precision_y;
        self.b += 0.5 * u_precision_u_zero;

        self.precision_u += &x_out_precision_y;
        self.mean = self.sigma.transform(&self.precision_u);

        let u_precision_u_n = frob_inner(&self.mean, &self.precision_u);
        self.b -= 0.5 * &u_precision_u_n;

        self.a += (self.t as f32) * (if downdate == true {-0.5} else {0.5});
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
        let mut precision_out = self.precision.clone();
        if (downdate) {
            self.precision_u -= &other.precision_u;
            precision_out -= &other.precision;
        } else {
            self.precision_u += &other.precision_u;
            precision_out += &other.precision;
        }

        self.sigma = precision_out.inverse();
        let mean_out : Array2<f32> = self.sigma.transform(&self.precision_u);

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = &other.mean - &mean_out;

        let u_diff_l_u_diff_one = self.precision.inner_product(&mean_one_diff, &mean_one_diff);
        let u_diff_l_u_diff_two = other.precision.inner_product(&mean_two_diff, &mean_two_diff);
        
        let s = if (downdate) {-1.0f32} else {1.0f32};
        
        self.b += s * (other.b + 0.5 * (u_diff_l_u_diff_one + u_diff_l_u_diff_two));

        self.a += s * (other.a + 0.5 * ((self.t * self.s) as f32));

        self.precision = precision_out;

        self.mean = mean_out;
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
