extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;


///Data point [input, output pair]
///with an output precision matrix
struct DataPoint {
    in_vec: Array1<f32>,
    out_vec: Array1<f32>,
    out_precision: Array2<f32>
}

///Normal-inverse-gamma distribution representation
///for bayesian inference
struct NormalInverseGamma {
    mean: Array2<f32>,
    precision_u: Array2<f32>,
    precision: Array4<f32>,
    sigma : Array4<f32>,
    a: f32,
    b: f32,
    t: usize,
    s: usize,
    proper: bool //Whether/not this has a defined sigma
}

///Allows doing dist ^= dist to invert dist in place
impl ops::BitXorAssign for NormalInverseGamma {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.precision_u *= -1.0;
        self.precision *= -1.0;
        self.sigma *= -1.0;
        self.a *= -1.0;
        self.a -= (self.t * self.s) as f32;
        self.b *= -1.0;
    }
}

fn invert_hermitian_array4(in_array: &Array4<f32>) -> Array4<f32> {
    let t = in_array.shape()[0];
    let s = in_array.shape()[1];
    let as_matrix: Array2<f32> = in_array.clone().into_shape((t * s, t * s)).unwrap();
    let as_matrix_inv: Array2<f32> = as_matrix.invh().unwrap();

    as_matrix_inv.into_shape((t, s, t, s)).unwrap()
}

impl NormalInverseGamma {
    fn update(&mut self, data_point : &DataPoint, downdate : bool) {
        let U = crate::linalg_utils::sqrtm(&data_point.out_precision);
        let out_precision = (if downdate == true {-1.0} else {1.0}) * &data_point.out_precision;
        
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


        let x_out_precision_y = einsum("s,tr,r->ts", 
                               &[&data_point.in_vec, &data_point.out_precision, &data_point.out_vec])
                                .unwrap().into_dimensionality::<Ix2>().unwrap();

        let y_T_out_precision_y = einsum("x,xy,y->", 
                               &[&data_point.out_vec, &data_point.out_precision, &data_point.out_vec])
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

impl ops::AddAssign for NormalInverseGamma {
    fn add_assign(&mut self, other: Self) {
        self.precision_u += &other.precision_u;

        let precision_out = &self.precision + &other.precision;

        self.sigma = invert_hermitian_array4(&precision_out);

        self.proper = true;

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
        s,
        proper: false
    }
}

