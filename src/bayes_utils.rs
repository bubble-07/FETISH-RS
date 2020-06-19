extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::cauchy_fourier_features::*;

use rand::prelude::*;
use rand_distr::{Cauchy, Distribution};
use rand_distr::StandardNormal;

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
    precision: Array4<f32>,
    sigma : Array4<f32>,
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

    pub fn sample(&self, rng : &mut ThreadRng) -> Array2::<f32> {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let vec_sample = self.sample_as_vec(rng);

        let result = vec_sample.into_shape((t, s)).unwrap();

        result
    }

    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];
        let std_norm_samp = gen_standard_normal_random(rng, t * s);
        let my_schmear : Schmear = tensors_to_schmear(&self.mean, &self.sigma);
        let mut result : Array1<f32> = einsum("ab,b->a", &[&my_schmear.covariance, &std_norm_samp])
                                         .unwrap().into_dimensionality::<Ix1>().unwrap();
        
        //Great, now we need to sample from the inverse-gamma part
        //to determine a multiplier for the covariance
        let inv_gamma_sample = gen_inverse_gamma_random(rng, self.a, self.b);
        result *= inv_gamma_sample;

        //Add the mean to offset it right
        result += &my_schmear.mean;
        result
    }

    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        mean_to_array(&self.mean)
    }
    pub fn get_schmear(&self) -> Schmear {
        let mut result = tensors_to_schmear(&self.mean, &self.sigma);
        result.covariance *= (self.a / self.b);
        result
    }
    pub fn get_inverse_schmear(&self) -> InverseSchmear {
        let mut result = tensors_to_inv_schmear(&self.mean, &self.precision);
        result.precision *= (self.b / self.a);
        result
    }
}

impl NormalInverseGamma {
    pub fn eval(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        einsum("ab,b->a", &[&self.mean, in_vec])
              .unwrap().into_dimensionality::<Ix1>().unwrap()
    }
}

impl NormalInverseGamma {
    pub fn new(mean : Array2<f32>, precision : Array4<f32>, a : f32, b : f32, t : usize, s : usize) -> NormalInverseGamma {
        let precision_u : Array2<f32> = einsum("abcd,cd->ab", &[&precision, &mean])
                                        .unwrap().into_dimensionality::<Ix2>().unwrap();
        let sigma = invert_hermitian_array4(&precision);
        
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

///Allows doing dist ^= dist to invert dist in place
impl ops::BitXorAssign<()> for NormalInverseGamma {
    fn bitxor_assign(&mut self, rhs: ()) {
        self.precision_u *= -1.0;
        self.precision *= -1.0;
        self.sigma *= -1.0;
        self.a *= -1.0;
        self.a -= (self.t * self.s) as f32;
        self.b *= -1.0;
    }
}

pub fn invert_hermitian_array4(in_array: &Array4<f32>) -> Array4<f32> {
    let t = in_array.shape()[0];
    let s = in_array.shape()[1];
    let as_matrix: Array2<f32> = in_array.clone().into_shape((t * s, t * s)).unwrap();
    let as_matrix_inv: Array2<f32> = as_matrix.invh().unwrap();

    as_matrix_inv.into_shape((t, s, t, s)).unwrap()
}

impl NormalInverseGamma {

    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let U = crate::linalg_utils::sqrtm(&data_point.out_inv_schmear.precision);
        let out_precision = (if downdate == true {-1.0} else {1.0}) * &data_point.out_inv_schmear.precision;
        
        let precision_contrib = einsum("ac,b,d->abcd", &[&out_precision, &data_point.in_vec, &data_point.in_vec])
                                .unwrap().into_dimensionality::<Ix4>().unwrap();

        self.precision += &precision_contrib;

        //Begin Woodbury formula inversion of sigma
        let sigma_x_U = einsum("abcd,d,ce->abe", &[&self.sigma, &data_point.in_vec, &U])
                                .unwrap();
        let x_T_U_sigma = einsum("a,bc,cade->bde", &[&data_point.in_vec, &U, &self.sigma])
                                .unwrap();
        let x_T_U_sigma_x_U = einsum("abc,c,bd->ad", &[&x_T_U_sigma, &data_point.in_vec, &U])
                                .unwrap().into_dimensionality::<Ix2>().unwrap();

        let Z = Array::eye(self.t) + (if downdate == true {-1.0} else {1.0}) * x_T_U_sigma_x_U;

        let Z_inv = Z.invh().unwrap();

        let sigma_diff = einsum("abe,ef,fcd", &[&sigma_x_U, &Z_inv, &x_T_U_sigma])
                                .unwrap().into_dimensionality::<Ix4>().unwrap();

        let sigma_diff_scaled = sigma_diff * (if downdate == true {1.0} else {-1.0});

        self.sigma += &sigma_diff_scaled;


        let data_out_mean : &Array1::<f32> = &data_point.out_inv_schmear.mean;

        let x_out_precision_y = einsum("s,tr,r->ts", 
                               &[&data_point.in_vec, &out_precision, data_out_mean])
                                .unwrap().into_dimensionality::<Ix2>().unwrap();

        let y_T_out_precision_y = einsum("x,xy,y->", 
                               &[data_out_mean, &out_precision, data_out_mean])
                                .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
        
        let u_precision_u_zero = einsum("ab,ab->", &[&self.mean, &self.precision_u])
                                .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();


        self.b += 0.5 * y_T_out_precision_y;
        self.b += 0.5 * u_precision_u_zero;

        self.precision_u += &x_out_precision_y;
        self.mean = einsum("abcd,cd->ab", &[&self.sigma, &self.precision_u])
                            .unwrap().into_dimensionality::<Ix2>().unwrap();

        let u_precision_u_n = einsum("ab,ab->", &[&self.mean, &self.precision_u])
                                .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
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

impl ops::AddAssign<&NormalInverseGamma> for NormalInverseGamma {
    fn add_assign(&mut self, other: &NormalInverseGamma) {
        self.precision_u += &other.precision_u;

        let precision_out = &self.precision + &other.precision;

        self.sigma = invert_hermitian_array4(&precision_out);

        let mean_out : Array2<f32> = einsum("abcd,cd->ab", &[&self.sigma, &self.precision_u])
                                        .unwrap().into_dimensionality::<Ix2>().unwrap();

        let mean_one_diff = &self.mean - &mean_out;
        let mean_two_diff = &other.mean - &mean_out;

        let u_diff_l_u_diff_one = einsum("ab,abcd,cd->", &[&mean_one_diff, &self.precision, &mean_one_diff])
                                        .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
        let u_diff_l_u_diff_two = einsum("ab,abcd,cd->", &[&mean_two_diff, &other.precision, &mean_two_diff])
                                        .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
        
        self.b += other.b + 0.5 * (u_diff_l_u_diff_one + u_diff_l_u_diff_two);

        self.a += other.a + 0.5 * ((self.t * self.s) as f32);

        self.precision = precision_out;

        self.mean = mean_out;
    }
}


fn zero_normal_inverse_gamma(t : usize, s : usize) -> NormalInverseGamma {
    NormalInverseGamma {
        mean: Array::zeros((t, s)),
        precision_u: Array::zeros((t, s)),
        precision: Array::zeros((t, s, t, s)),
        sigma: Array::zeros((t, s, t, s)), //I know this is invalid
        a: ((t * s) as f32) * -0.5f32,
        b: 0.0,
        t,
        s
    }
}

