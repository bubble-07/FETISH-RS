extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
use crate::linalg_utils::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::closest_psd_matrix::*;
use crate::randomized_svd::*;
use crate::schmear::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
pub struct FuncScatterTensor {
    pub in_scatter : Array2<f32>,
    pub out_scatter : Array2<f32>,
    pub scale : f32
}

impl FuncScatterTensor {
    pub fn from_in_and_out_scatter(in_scatter : Array2<f32>, out_scatter : Array2<f32>) -> FuncScatterTensor {
        let mut result = FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale : 1.0f32
        };
        result.renormalize();
        result
    }
    pub fn from_four_tensor(in_tensor : &Array4<f32>) -> FuncScatterTensor {
        //Takes the t x s x t x s four-dimensional tensor
        //representation and finds the best tensor-product
        //representation
        let t = in_tensor.shape()[0];
        let s = in_tensor.shape()[1];
        
        //first re-arrange to be t x t x s x s
        let re_arranged = einsum("abcd->acbd", &[in_tensor]).unwrap();
        //Now re-shape to be (t * t) x (s * s)
        let reshaped = re_arranged.into_shape((t * t, s * s)).unwrap()
                       .into_dimensionality::<Ix2>().unwrap();
        
        let mut rng = rand::thread_rng();
        let (u, scale, v) = randomized_rank_one_approx(&reshaped, &mut rng);

        //Great, now we just need to reshape and find the nearest PSD matrices
        let reshaped_u = u.into_shape((t, t)).unwrap()
                          .into_dimensionality::<Ix2>().unwrap();
        let reshaped_v = v.into_shape((s, s)).unwrap()
                          .into_dimensionality::<Ix2>().unwrap();

        let in_scatter = get_closest_unit_norm_psd_matrix(&reshaped_u);
        let out_scatter = get_closest_unit_norm_psd_matrix(&reshaped_v);

        FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale
        }
    }
    pub fn to_tensor4(&self) -> Array4<f32> {
        let mut result = einsum("ac,bd->abcd", &[&self.out_scatter, &self.in_scatter]).unwrap()
                               .into_dimensionality::<Ix4>().unwrap();
        result *= self.scale;
        result
    }
    pub fn flatten(&self) -> Array2<f32> {
        let t = self.out_scatter.shape()[0];
        let s = self.in_scatter.shape()[0];
        let result = self.to_tensor4().into_shape((t * s, t * s)).unwrap();
        result
    }

    pub fn update_to_inverse_woodbury(&mut self, in_vec : &Array1<f32>, out_precision : &Array2<f32>,
                                      downdate : bool) {
        let t : usize = out_precision.shape()[0];
        let s : usize = in_vec.shape()[0];

        let U = sqrtm(&out_precision);

        let in_scatter_in_vec : Array1<f32> = einsum("ab,b->a", &[&self.in_scatter, in_vec]).unwrap()
                                              .into_dimensionality::<Ix1>().unwrap();

        let out_scatter_U : Array2<f32> = einsum("ab,bc->ac", &[&self.out_scatter, &U]).unwrap()
                                              .into_dimensionality::<Ix2>().unwrap();

        let mut x_T_U_sigma = einsum("db,e->bde", &[&out_scatter_U, &in_scatter_in_vec]).unwrap();
        x_T_U_sigma *= self.scale;
        

        let x_T_U_sigma_x_U = einsum("abc,c,bd->ad", &[&x_T_U_sigma, in_vec, &U])
                                .unwrap().into_dimensionality::<Ix2>().unwrap();

        let Z = Array::eye(t) + (if downdate == true {-1.0} else {1.0}) * x_T_U_sigma_x_U;
        let Z_inv = Z.invh().unwrap();
        
        let out_diff = einsum("ae,ef,cf->ac", &[&out_scatter_U, &Z_inv, &out_scatter_U]).unwrap()
                             .into_dimensionality::<Ix2>().unwrap();
        let in_diff = einsum("b,d->bd", &[&in_scatter_in_vec, &in_scatter_in_vec]).unwrap()
                             .into_dimensionality::<Ix2>().unwrap();
        let scale_diff = self.scale * self.scale;


        let mut total_diff = FuncScatterTensor {
            in_scatter : in_diff,
            out_scatter : out_diff,
            scale : scale_diff
        };
        total_diff.renormalize();
        //"downdate" is inverted here, because in the woodbury formula, there's a minus sign on the
        //delta to the inverse matrix
        self.update(&total_diff, !downdate);
    }

    ///Transform a t x s mean matrix
    pub fn transform(&self, mean : &Array2<f32>) -> Array2<f32> {
        let mut result : Array2<f32> = einsum("ab,cd,bd->ac", &[&self.out_scatter, &self.in_scatter, mean]).unwrap()
                                   .into_dimensionality::<Ix2>().unwrap();
        result *= self.scale;
        result
    }

    ///Transform a t x s x m matrix to another t x s x m one
    pub fn transform3(&self, tensor : &Array3<f32>) -> Array3<f32> {
        let mut result : Array3<f32> = einsum("ab,cd,bds->acs", &[&self.out_scatter, &self.in_scatter, tensor]).unwrap()
                                   .into_dimensionality::<Ix3>().unwrap();
        result *= self.scale;
        result
    }

    ///Induced inner product on t x s mean matrices
    pub fn inner_product(&self, mean_one : &Array2<f32>, mean_two : &Array2<f32>) -> f32 {
        let result = einsum("ab,ac,bd,cd->", &[mean_one, &self.out_scatter, &self.in_scatter, mean_two])
                           .unwrap().into_dimensionality::<Ix0>().unwrap().into_scalar();
        result * self.scale
    }

    pub fn transform_in_out(&self, in_array : &Array2<f32>) -> Array2<f32> {
        let in_inner = einsum("ab,ab->", &[&self.in_scatter, in_array]).unwrap()
                       .into_dimensionality::<Ix0>().unwrap().into_scalar();
        let mut out = self.out_scatter.clone();
        out *= in_inner * self.scale;
        out
    }

    ///Gets the matrix sqrt of this
    pub fn sqrt(&self) -> FuncScatterTensor {
        let in_scatter = sqrtm(&self.in_scatter);
        let out_scatter = sqrtm(&self.out_scatter);
        let scale = self.scale.sqrt();

        let mut result = FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale
        };
        result.renormalize();
        result
    }

    pub fn inverse(&self) -> FuncScatterTensor {
        let inv_in_scatter = self.in_scatter.invh().unwrap();
        let inv_out_scatter = self.out_scatter.invh().unwrap();
        let inv_scale = 1.0f32 / self.scale;
        
        let mut result = FuncScatterTensor {
            in_scatter : inv_in_scatter,
            out_scatter : inv_out_scatter,
            scale : inv_scale
        };
        result.renormalize();
        result
    }
    fn renormalize(&mut self) {
        let in_scatter_norm = self.in_scatter.opnorm_fro().unwrap();
        let out_scatter_norm = self.out_scatter.opnorm_fro().unwrap();

        self.scale *= in_scatter_norm * out_scatter_norm;
        self.in_scatter /= in_scatter_norm;
        self.out_scatter /= out_scatter_norm;
    }

    //Uses the approximate rank-1 approx to the sum of rank-1 matrices
    //that you found
    fn update(&mut self, other : &FuncScatterTensor, downdate : bool) {
        let tot_scale_sq = self.scale * self.scale + other.scale * other.scale;
        let tot_scale = tot_scale_sq.sqrt();
        //First, scale your elements
        self.in_scatter *= self.scale;
        self.out_scatter *= self.scale;
        //Add scaled versions of the other elems
        if (downdate) {
            //TODO: is there a better approx to the downdate than doing this?
            self.in_scatter -= &(other.scale * &other.in_scatter);
            self.out_scatter -= &(other.scale * &other.out_scatter);
        } else {
            self.in_scatter += &(other.scale * &other.in_scatter);
            self.out_scatter += &(other.scale * &other.out_scatter);
        }
        //Divide through by the total scale
        self.scale = 1.0f32 / tot_scale;
        //Renormalize to put things back into standard form
        self.renormalize();
    }
}

impl ops::MulAssign<f32> for FuncScatterTensor {
    fn mul_assign(&mut self, scale_fac : f32) {
        self.scale *= scale_fac;
    }
}

impl ops::AddAssign<&FuncScatterTensor> for FuncScatterTensor {
    fn add_assign(&mut self, other : &FuncScatterTensor) {
        self.update(other, false);
    }
}

impl ops::SubAssign<&FuncScatterTensor> for FuncScatterTensor {
    fn sub_assign(&mut self, other : &FuncScatterTensor) {
        self.update(other, true);
    }
}
