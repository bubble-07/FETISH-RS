extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use ndarray::*;
use crate::array_utils::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use noisy_float::prelude::*;

use rand::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::ChiSquared;

use crate::test_utils::*;
use crate::inverse_schmear::*;
use crate::schmear::*;
use crate::linalg_utils::*;
use crate::func_scatter_tensor::*;

pub struct FuncSchmear {
    pub mean : Array2<f32>,
    pub covariance : FuncScatterTensor
}

impl FuncSchmear {
    //Flattens and then compresses this func schmear by the given
    //projection mat
    pub fn compress(&self, mat : &Array2<f32>) -> Schmear {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let mean_flat = self.mean.clone().into_shape((t * s,)).unwrap();
        let mean_transformed = mat.dot(&mean_flat);

        let covariance_transformed  = self.covariance.compress(mat);

        Schmear {
            mean : mean_transformed,
            covariance : covariance_transformed
        }
    }

    pub fn flatten(&self) -> Schmear {
        let t = self.mean.shape()[0];
        let s = self.mean.shape()[1];

        let mean = self.mean.clone().into_shape((t * s,)).unwrap();
        let covariance = self.covariance.flatten();
        Schmear {
            mean,
            covariance
        }
    }
    ///Computes the output schmear of this func schmear applied
    ///to a given argument schmear
    pub fn apply(&self, x : &Schmear) -> Schmear {
        let sigma_dot_u = frob_inner(&self.covariance.in_scatter, &x.covariance);
        let u_inner_product = x.mean.dot(&self.covariance.in_scatter).dot(&x.mean);
        let v_scale = sigma_dot_u + u_inner_product;
        let v_contrib = v_scale * &self.covariance.out_scatter;

        if (v_scale < 0.0f32) {
            println!("v scale became negative: {}", v_scale);
            println!("components: {}, {}",  sigma_dot_u, u_inner_product);
            if (u_inner_product < 0.0f32) {
                println!("Non-psd in scatter: {}", &self.covariance.in_scatter);
                println!("x mean: {}", &x.mean);
            }
        }

        let m_sigma_m_t = self.mean.dot(&x.covariance).dot(&self.mean.t());

        let result_covariance = v_contrib + &m_sigma_m_t;
        let result_mean = self.mean.dot(&x.mean);
        let result = Schmear {
            mean : result_mean,
            covariance : result_covariance
        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schmear_application_accurate() {
        let t = 3;
        let s = 3;
        let num_samps = 1000;
        let func_scale_mult = 0.01f32;
        let arg_scale_mult = 0.01f32;

        let normal_inverse_wishart = random_normal_inverse_wishart(s, t);

        let func_schmear = normal_inverse_wishart.get_schmear();

        let arg_mean = random_vector(s);
        let mut arg_covariance_sqrt = random_matrix(s, s);
        arg_covariance_sqrt *= arg_scale_mult; 
        let arg_covariance = arg_covariance_sqrt.dot(&arg_covariance_sqrt.t());
        let arg_schmear = Schmear {
            mean : arg_mean.clone(),
            covariance : arg_covariance.clone()
        };

        let actual_out_schmear = func_schmear.apply(&arg_schmear);

        let mut expected_out_mean = Array::zeros((t,));
        let mut expected_out_covariance = Array::zeros((t, t));

        let mut rng = rand::thread_rng();

        let scale_fac = 1.0f32 / (num_samps as f32);

        for _ in 0..num_samps {
            let func_samp = normal_inverse_wishart.sample(&mut rng);

            let func_diff = &func_samp - &func_schmear.mean;

            let standard_normal_arg_vec = Array::random((s,), StandardNormal);
            let arg_samp = &arg_mean + &arg_covariance_sqrt.dot(&standard_normal_arg_vec);

            let out_samp = func_samp.dot(&arg_samp);

            let out_diff = &out_samp - &actual_out_schmear.mean;

            expected_out_mean += &(scale_fac * &out_samp);
            expected_out_covariance += &(scale_fac * &outer(&out_diff, &out_diff));
        }

        assert_equal_vectors_to_within(&actual_out_schmear.mean, &expected_out_mean, 1.0f32);
        assert_equal_matrices_to_within(&actual_out_schmear.covariance, &expected_out_covariance, 10.0f32);
    }
}

