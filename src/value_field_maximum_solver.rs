extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use anyhow::*;

use crate::array_utils::*;
use crate::inverse_schmear::*;
use crate::type_id::*;
use crate::space_info::*;
use crate::value_field::*;
use argmin::prelude::*;

pub struct ValueFieldMaximumSolver {
    pub func_mat : Array2<f32>,
    pub value_field : ValueField
}

impl ValueFieldMaximumSolver {
    pub fn get_compressed_vector_with_max_value<'a>(&self, compressed_vecs : &'a Vec<Array1<f32>>) -> &'a Array1<f32> { 
        let mut max_value = f32::NEG_INFINITY;
        let mut result = &compressed_vecs[0];
        for compressed_vec in compressed_vecs {
            let value = self.get_value(compressed_vec);
            if (value > max_value) {
                max_value = value;
                result = compressed_vec;
            }
        }
        result
    }

    pub fn get_type_id(&self) -> TypeId {
        self.value_field.get_type_id()
    }

    pub fn get_value(&self, x_compressed : &Array1<f32>) -> f32 {
        self.value_field.get_value_for_vector(x_compressed)
    }

    pub fn get_value_gradient(&self, x_compressed : &Array1<f32>) -> Array1<f32> {
        let func_space_info = get_function_space_info(self.get_type_id());
        let y_compressed = func_space_info.apply(&self.func_mat, x_compressed);

        let func_jacobian = func_space_info.jacobian(&self.func_mat, x_compressed);
        let out_feat_jacobian = func_space_info.out_feat_info.get_feature_jacobian(&y_compressed);

        let value_field_grad = self.value_field.coefs.dot(&out_feat_jacobian).dot(&func_jacobian);

        let compressed_inv_schmear = &self.value_field.prior_schmear.compressed_inv_schmear;
        let diff = &y_compressed - &compressed_inv_schmear.mean;
        let precision_func_jacobian = compressed_inv_schmear.precision.dot(&func_jacobian);
        let delta = diff.dot(&precision_func_jacobian);
        let sq_mahalanobis_grad = 2.0f32 * &delta;

        let mut result = value_field_grad;
        result -= &sq_mahalanobis_grad;
        result
    }
}

impl ArgminOp for ValueFieldMaximumSolver {
    type Param = Array1<f32>;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, x_compressed : &Self::Param) -> Result<Self::Output, Error> {
        let neg_result = self.get_value(x_compressed);
        let result = -neg_result;
        if (result.is_finite()) {
            Ok(result)
        } else {
            Err(anyhow!("Non-finite value for vector: {}", x_compressed))
        }
    }

    fn gradient(&self, x_compressed : &Self::Param) -> Result<Self::Param, Error> {
        let mut neg_result = self.get_value_gradient(x_compressed);
        neg_result *= -1.0f32;
        if (all_finite(&neg_result)) {
            Ok(neg_result)
        } else {
            Err(anyhow!("Non-finite gradient {} for vector {}", neg_result, x_compressed))
        }
    }
}
