extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use ndarray_einsum_beta::*;
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
