extern crate ndarray;
extern crate ndarray_linalg;

use std::ops;
use std::cmp;
use ndarray::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::*;
use crate::closest_psd_matrix::*;
use crate::test_utils::*;
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


        //Start off by decomposing the covariance into its eigendecomposition
        let maybe_eigh = covariance.eigh(UPLO::Lower);
        if let Result::Err(e) = &maybe_eigh {
            error!("Bad matrix for eigh {}", covariance);
        }
        //Time complexity of the loop: nts^2 [so long as n is O(ln(ts)) and s >> t [wlog]
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

                in_scatter += &(coef * outer(&in_vec, &in_vec));
                out_scatter += &(coef * &outer(&out_vec, &out_vec));
            }
        }
        //Now, we need to compute the appropriate scale by computing
        //<v, w> / <v, v>, where w is the actual tensor, and v is our estimate
        let expansion = linear_sketch.get_expansion_matrix();
        let n = expansion.shape()[1];
        //t x (s x n)
        let expanded_expansion = expansion.into_shape((t, s, n)).unwrap();

        //s x (t x n)
        let permuted_expansion = expanded_expansion.permuted_axes([1, 0, 2]);

        let permuted_expansion_std = permuted_expansion.as_standard_layout();

        //s x (t * n)
        let reshaped_expansion = permuted_expansion_std.into_shape((s, t * n)).unwrap();

        //s x (t * n)
        let transformed_expansion = in_scatter.dot(&reshaped_expansion);
        //(t * n) x s
        let reshaped_expansion_t = reshaped_expansion.t();

        //(t * n) x (t * n)
        let reduced_expansion = reshaped_expansion_t.dot(&transformed_expansion);

        //t x n x t x n
        let reduced_expansion_4d = reduced_expansion.into_shape((t, n, t, n)).unwrap();

        //n x n x t x t
        let rearranged_reduced_4d = reduced_expansion_4d.permuted_axes([1, 3, 0, 2]);
        let rearranged_reduced_4d_std = rearranged_reduced_4d.as_standard_layout();

        //(n * n) x (t * t)
        let rearranged_reduced = rearranged_reduced_4d_std.into_shape((n * n, t * t)).unwrap();

        let out_scatter_flat = out_scatter.clone().into_shape((t * t,)).unwrap();
        let covariance_flat = covariance.clone().into_shape((n * n,)).unwrap();

        //(n * n)
        let double_reduced = rearranged_reduced.dot(&out_scatter_flat);

        let dot = double_reduced.dot(&covariance_flat);
        let sq_norm = double_reduced.dot(&double_reduced);

        let scale = dot / sq_norm;

        let mut result = FuncScatterTensor {
            in_scatter,
            out_scatter,
            scale
        };
        result.renormalize();
        result
    }

    pub fn flatten(&self) -> Array2<f32> {
        let result = self.scale * kron(&self.out_scatter, &self.in_scatter);
        result
    }

    ///Transform a t x s mean matrix
    pub fn transform(&self, mean : &Array2<f32>) -> Array2<f32> {
        let mean_in_scatter : Array2<f32> = mean.dot(&self.in_scatter);
        let mut result = self.out_scatter.dot(&mean_in_scatter);
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
        let other_scale = if (downdate) { -other.scale } else { other.scale };
        
        let in_dot = frob_inner(&self.in_scatter, &other.in_scatter);
        let out_dot = frob_inner(&self.out_scatter, &other.out_scatter);

        let tot_scale_sq = self.scale * self.scale + 
                           other_scale * other_scale + 
                           2.0f32 * self.scale * other_scale * in_dot * out_dot;

        let tot_scale = tot_scale_sq.sqrt();

        //First, scale your elements
        self.in_scatter *= self.scale;
        self.out_scatter *= self.scale;

        //Add scaled versions of the other elems
        self.in_scatter += &(other_scale * &other.in_scatter);
        self.out_scatter += &(other_scale * &other.out_scatter);

        if (tot_scale < ZEROING_THRESH) {
            panic!(); //Awwww, freak out!
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse() {
        let t = 5;
        let s = 6;
        let mat = random_matrix(t, s);
        let scatter_tensor = random_func_scatter_tensor(t, s);
        let scatter_tensor_inv = scatter_tensor.inverse();

        let transformed = scatter_tensor.transform(&mat);
        let actual = scatter_tensor_inv.transform(&transformed);

        assert_equal_matrices(&actual, &mat);

        let post_actual = scatter_tensor.transform(&actual);
        assert_equal_matrices(&post_actual, &transformed);
    }

    #[test]
    fn test_sqrt() {
        let t = 5;
        let s = 6;
        let mat = random_matrix(t, s);
        let scatter_tensor = random_func_scatter_tensor(t, s);
        let scatter_tensor_sqrt = scatter_tensor.sqrt(); 

        let expected = scatter_tensor.transform(&mat);

        let half_actual = scatter_tensor_sqrt.transform(&mat);
        let actual = scatter_tensor_sqrt.transform(&half_actual);

        assert_equal_matrices(&actual, &expected);
    }

    #[test]
    fn test_rank_one_from_compressed_covariance() {
        let t = 10;
        let s = 10;
        let func_scatter_tensor = random_func_scatter_tensor(t, s);
        let covariance = func_scatter_tensor.flatten();
        let linear_sketch = LinearSketch::trivial_sketch(t * s);
        let from_compressed = FuncScatterTensor::from_compressed_covariance(t, s, &linear_sketch, &covariance);

        assert_eps_equals(from_compressed.scale, func_scatter_tensor.scale);
        assert_equal_matrices(&from_compressed.in_scatter, &func_scatter_tensor.in_scatter);
        assert_equal_matrices(&from_compressed.out_scatter, &func_scatter_tensor.out_scatter);
    }

    #[test]
    fn plot_relative_error_against_true_value() {
        let samples = 100;
        let buckets = 10;
        let t = 10;
        let s = 10;
        let mut relative_errors = Vec::<f64>::new();
        for _ in 0..samples {
            let one = random_func_scatter_tensor(t, s);
            let two = random_func_scatter_tensor(t, s);
            let mut actual_sum = one.clone();
            actual_sum += &two;

            let actual_sum_flat = actual_sum.flatten();

            let one_flat = one.flatten();
            let two_flat = two.flatten();
            let mut expected_sum_flat = one_flat.clone();
            expected_sum_flat += &two_flat;

            let relative_error = relative_frob_norm_error(&actual_sum_flat, &expected_sum_flat);
            relative_errors.push(relative_error as f64);
        }
        plot_histogram("scatterTensorRelativeErrorAgainstTrueValue", relative_errors, buckets);
    }

    #[test]
    fn adding_same_tensor_increasing_in_same_subspace() {
        let tensor = random_func_scatter_tensor(10, 9);
        let mut doubled_tensor = tensor.clone();
        doubled_tensor += &tensor;
        assert_greater(doubled_tensor.scale, tensor.scale);
        assert_equal_matrices(&doubled_tensor.in_scatter, &tensor.in_scatter);
        assert_equal_matrices(&doubled_tensor.out_scatter, &tensor.out_scatter);
    }
    #[test]
    fn subtracting_half_tensor_decreasing_in_same_subspace() {
        let tensor = random_func_scatter_tensor(10, 9);
        let mut halved_tensor = tensor.clone();
        halved_tensor *= 0.5f32;
        let mut subtracted_tensor = tensor.clone();
        subtracted_tensor -= &halved_tensor;

        assert_greater(tensor.scale, halved_tensor.scale);
        assert_equal_matrices(&halved_tensor.in_scatter, &tensor.in_scatter); 
        assert_equal_matrices(&halved_tensor.out_scatter, &tensor.out_scatter);

        assert_greater(tensor.scale, subtracted_tensor.scale);
        assert_equal_matrices(&subtracted_tensor.in_scatter, &tensor.in_scatter);
        assert_equal_matrices(&subtracted_tensor.out_scatter, &tensor.out_scatter);

        assert_eps_equals(halved_tensor.scale, subtracted_tensor.scale);
    }

}
