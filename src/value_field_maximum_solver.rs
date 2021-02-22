extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;


use std::ops;
use std::rc::*;

use crate::function_space_info::*;
use crate::feature_space_info::*;
use crate::data_points::*;
use crate::sigma_points::*;
use crate::embedder_state::*;
use crate::pseudoinverse::*;
use crate::term_pointer::*;
use crate::normal_inverse_wishart::*;
use crate::alpha_formulas::*;
use crate::vector_space::*;
use crate::feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::enum_feature_collection::*;
use crate::func_scatter_tensor::*;
use crate::linalg_utils::*;
use crate::linear_sketch::*;
use crate::model::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::inverse_schmear::*;
use crate::func_inverse_schmear::*;
use crate::data_point::*;
use rand::prelude::*;
use argmin::prelude::*;

pub struct ValueFieldMaximumSolver {
    pub func_space_info : FunctionSpaceInfo,
    pub func_mat : Array2<f32>,
    pub value_field_coefs : Array1<f32>,
    pub target_compressed_inv_schmear : Option<InverseSchmear>
}

impl ArgminOp for ValueFieldMaximumSolver {
    type Param = Array1<f32>;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, x_compressed : &Self::Param) -> Result<Self::Output, Error> {
        let y_compressed = self.func_space_info.apply(&self.func_mat, x_compressed);
        
        let sq_mahalanobis = match (&self.target_compressed_inv_schmear) {
            Option::None => 0.0f32,
            Option::Some(compressed_inv_schmear) => compressed_inv_schmear.sq_mahalanobis_dist(&y_compressed)
        };

        let y_feats = self.func_space_info.out_feat_info.get_features(&y_compressed);
        let value_field_eval = self.value_field_coefs.dot(&y_feats);

        Ok(-(value_field_eval + -sq_mahalanobis))
    }

    fn gradient(&self, x_compressed : &Self::Param) -> Result<Self::Param, Error> {
        let y_compressed = self.func_space_info.apply(&self.func_mat, x_compressed);

        let func_jacobian = self.func_space_info.jacobian(&self.func_mat, x_compressed);
        let out_feat_jacobian = self.func_space_info.out_feat_info.get_feature_jacobian(&y_compressed);

        let value_field_grad = self.value_field_coefs.dot(&out_feat_jacobian).dot(&func_jacobian);

        let sq_mahalanobis_grad = match (&self.target_compressed_inv_schmear) {
            Option::None => Array::zeros((value_field_grad.shape()[0],)),
            Option::Some(compressed_inv_schmear) => {
                let diff = &y_compressed - &compressed_inv_schmear.mean;
                let precision_func_jacobian = compressed_inv_schmear.precision.dot(&func_jacobian);
                let delta = diff.dot(&precision_func_jacobian);
                let result = 2.0f32 * &delta;
                result
            }
        };
        let mut result = value_field_grad;
        result *= -1.0f32;
        result += &sq_mahalanobis_grad;
        Ok(result)
    }
}
