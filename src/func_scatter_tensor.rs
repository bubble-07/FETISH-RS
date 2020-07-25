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
use crate::pseudoinverse::*;
use crate::params::*;

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
        let mut re_arranged = in_tensor.clone();
        re_arranged.swap_axes(1, 2);

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

        let out_scatter = get_closest_unit_norm_psd_matrix(&reshaped_u);
        let in_scatter = get_closest_unit_norm_psd_matrix(&reshaped_v);

        FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale
        }
    }
    //pub fn to_tensor4(&self) -> Array4<f32> {
    //    let mut result = einsum("ac,bd->abcd", &[&self.out_scatter, &self.in_scatter]).unwrap()
    //                           .into_dimensionality::<Ix4>().unwrap();
    //    result *= self.scale;
    //    result
    //}
    pub fn flatten(&self) -> Array2<f32> {
        let result = kron(&self.out_scatter, &self.in_scatter);
        result
    }

    ///Transform a t x s mean matrix
    pub fn transform(&self, mean : &Array2<f32>) -> Array2<f32> {
        let mean_in_scatter : Array2<f32> = mean.dot(&self.in_scatter);
        let mut result = self.out_scatter.dot(&mean_in_scatter);
        result *= self.scale;
        result
    }

    ///Transform a t x s x m matrix to another t x s x m one
    pub fn transform3(&self, tensor : &Array3<f32>) -> Array3<f32> {
        let mut result : Array3<f32> = einsum("ca,abs,bd->cds", &[&self.out_scatter, tensor, &self.in_scatter]).unwrap()
                                   .into_dimensionality::<Ix3>().unwrap();
        result *= self.scale;
        result
    }

    ///Induced inner product on t x s mean matrices
    pub fn inner_product(&self, mean_one : &Array2<f32>, mean_two : &Array2<f32>) -> f32 {
        let transformed = self.transform(mean_two);
        let result = frob_inner(mean_one, &transformed);
        result
    }

    pub fn transform_in_out(&self, in_array : &Array2<f32>) -> Array2<f32> {
        let in_inner = frob_inner(&self.in_scatter, in_array);
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
        let inv_in_scatter = pseudoinverse_h(&self.in_scatter);
        let inv_out_scatter = pseudoinverse_h(&self.out_scatter);
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
        let combined_norm = in_scatter_norm * out_scatter_norm;
        if (combined_norm < ZEROING_THRESH) {
            //To smol to matter
            return;
        }

        self.scale *= combined_norm;
        self.in_scatter /= in_scatter_norm;
        self.out_scatter /= out_scatter_norm;
    }

    //Uses the approximate rank-1 approx to the sum of rank-1 matrices
    //that you found
    fn update(&mut self, other : &FuncScatterTensor, downdate : bool) {
        let tot_scale_sq = self.scale * self.scale + other.scale * other.scale;
        let tot_scale = tot_scale_sq.sqrt();
        if (tot_scale < ZEROING_THRESH) {
            return; //Too smol to matter
        }

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
