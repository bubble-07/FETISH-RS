extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::schmear::*;
use crate::data_point::*;
use crate::wishart::*;
use crate::inverse_schmear::*;
use crate::cauchy_fourier_features::*;

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
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
    }

    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1<f32> {
    }

    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        mean_to_array(&self.mean)
    }
    pub fn get_mean(&self) -> Array2<f32> {
        self.mean.clone()
    }

    pub fn get_schmear(&self) -> Schmear {
    }
    pub fn get_inverse_schmear(&self) -> InverseSchmear {
    }
    pub fn get_precision(&self) -> Array4<f32> {
    }

    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        einsum("ab,b->a", &[&self.mean, in_vec])
              .unwrap().into_dimensionality::<Ix1>().unwrap()
    }
}

impl NormalInverseWishart {
    pub fn new(mean : Array2<f32>, precision : Array2<f32>, big_v : Array2<f32>, little_v : f32) -> NormalInverseWishart {
        let precision_u : Array2<f32> = einsum("as,ts->at", &[&precision, &mean])
                                        .unwrap().into_dimensionality::<Ix2>().unwrap();
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
        self.little_v *= -1.0;
        self.little_v -= 2.0f32 * ((self.t) as f32);
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
        little_v: -(t as f32),
        t,
        s
    }
}

impl NormalInverseWishart {
    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let w = if (downdate) { -data_point.weight } else { data_point.weight };

        let mut w_x_x = einsum("a,b->ab", &[&data_point.in_vec, &data_point.in_vec]).unwrap();
        w_x_x *= w;

        let mut w_y_x_and_precision_u = einsum("a,b->ab", &[&data_point.out_vec, &data_point.in_vec]).unwrap();
        w_y_x_and_precision_u *= w;

        let mut w_y_y = einsum("a,b->ab", &[&data_point.out_vec, &data_point.out_vec]).unwrap();
        w_y_y *= w;

        self.precision += &w_x_x;



        //Use the woodbury matrix formula to update sigma
        let sigma_x = einsum("ab,b->a", &[&self.sigma, &data_point.in_vec]).unwrap();

        let mut w_over_one_plus_w_x_sigma_x : f32 = einsum("a,a->", &[&data_point.in_vec, &sigma_x]).unwrap()
                                                    .into_dimensionality::<Ix0>().unwrap().into_scalar();
        w_over_one_plus_w_x_sigma_x *= w;
        w_over_one_plus_w_x_sigma_x += 1.0f32;
        w_over_one_plus_w_x_sigma_x = w / w_over_one_plus_w_x_sigma_x;

        let mut rank_one_sigma_update = einsum("a,b->ab", &[&sigma_x, &sigma_x]).unwrap();
        rank_one_sigma_update *= w_over_one_plus_w_x_sigma_x;

        self.sigma -= &rank_one_sigma_update;


        let u_precision_u_zero = einsum("ts,vs->tv", &[&self.mean, &self.precision_u]).unwrap();


        w_y_x_and_precision_u += &self.precision_u;
        self.mean = einsum("ab,ta->tb", &[&self.sigma, &w_y_x_and_precision_u]).unwrap()
                          .into_dimensionality::<Ix2>().unwrap();


        self.little_v += if (downdate) {-1.0f32} else {1f32};

        
        self.precision_u = einsum("ts,sa->ta", &[&self.mean, &self.precision]).unwrap()
                           .into_dimensionality::<Ix2>().unwrap();

        self.big_v += &w_y_y;
        self.big_v += &u_precision_u_zero;
        self.big_v -= &einsum("ts,vs->tv", &[&self.mean, &self.precision_u]).unwrap();
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

impl ops::AddAssign<&NormalInverseWishart> for NormalInverseWishart {
    fn add_assign(&mut self, other: &NormalInverseWishart) {
        self.precision_u += &other.precision_u;

        let precision_out = &self.precision + &other.precision;

        self.sigma = precision_out.invh().unwrap();


        let mean_out : Array2<f32> = einsum("sa,ta->ts", &[&self.sigma, &self.precision_u])
                                        .unwrap().into_dimensionality::<Ix2>().unwrap();

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = &other.mean - &mean_out;



        let u_diff_l_u_diff_one = einsum("ta,ab,vb->tv", &[&mean_one_diff, &self.precision, &mean_one_diff])
                                        .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
        let u_diff_l_u_diff_two = einsum("ta,ab,vb->tv->", &[&mean_two_diff, &other.precision, &mean_two_diff])
                                        .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();

        self.little_v += other.little_v + (self.t as f32);
        self.big_v += &other.big_v;
        self.big_v += u_diff_l_u_diff_one;
        self.big_v += u_diff_l_u_diff_two;

        self.precision = precision_out;

        self.mean = mean_out;
    }
}


