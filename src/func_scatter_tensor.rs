extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use std::cmp;
use ndarray::*;
use ndarray_einsum_beta::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
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
    pub fn from_compressed_covariance(t : usize, s : usize, linear_sketch : &LinearSketch, 
                                                            covariance : &Array2<f32>) -> FuncScatterTensor {
        let mut in_scatter : Array2<f32> = Array::zeros((s, s));
        let mut out_scatter : Array2<f32> = Array::zeros((t, t));

        let mut accum_norm_sq = 0.0f32;

        //Start off by decomposing the covariance into its eigendecomposition
        let maybe_eigh = covariance.eigh(UPLO::Lower);
        if let Result::Err(e) = &maybe_eigh {
            error!("Bad matrix for eigh {}", covariance);
        }
        let (eigenvals, eigenvecs) = maybe_eigh.unwrap();
        for i in 0..eigenvecs.shape()[1] {
            let eigenval = eigenvals[i];
            let eigenvec = eigenvecs.column(i);
            //Now, for each eigenvector, expand it to the full size (t x s)
            let expanded_eigenvec = linear_sketch.expand(&eigenvec.to_owned());
            let reshaped = expanded_eigenvec.into_shape((t, s)).unwrap();

            //For each matrix of this form, obtain its SVD
            let maybe_svd = reshaped.svd(true, true); 
            if let Result::Err(e) = &maybe_svd {
                error!("Bad matrix for svd {}", reshaped);
            }
            let (maybe_u, sigma, maybe_v_t) = maybe_svd.unwrap();
            let u = maybe_u.unwrap();
            let v_t = maybe_v_t.unwrap();
            let n = cmp::min(cmp::min(s, t), sigma.shape()[0]);
            for j in 0..n {
                let out_vec = u.column(j).to_owned();
                let in_vec = v_t.row(j).to_owned();
                let singular_val = sigma[[j,]];

                let coef = singular_val * eigenval;
                accum_norm_sq += coef * coef;

                in_scatter += &(coef * outer(&in_vec, &in_vec));
                out_scatter += &(coef * &outer(&out_vec, &out_vec));
            }
        }
        let scale = 1.0f32 / accum_norm_sq.sqrt();
        let mut result = FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale
        };
        result.renormalize();
        result
    }

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
        println!("Transform3 tensor dims: {}, {}, {}", tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        println!("scatter dims: {}, {}", self.out_scatter.shape()[0], self.in_scatter.shape()[0]);
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
