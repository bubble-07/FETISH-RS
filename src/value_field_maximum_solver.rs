extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use anyhow::*;

use crate::sampled_value_field::*;
use crate::value_field::*;
use argmin::prelude::*;
use fetish_lib::everything::*;
use fetish_lib::everything::Context;

pub struct ValueFieldMaximumSolver<'a> {
    pub func_mat : Array2<f32>,
    pub value_field : SampledValueField<'a>,
    pub func_type_id : TypeId
}

impl<'a> ValueFieldMaximumSolver<'a> {
    fn get_context(&self) -> &Context {
        self.value_field.get_context()
    }
    pub fn get_compressed_vector_with_max_value<'b>(&self, compressed_vecs : &'b Vec<Array1<f32>>) -> &'b Array1<f32> { 
        let mut max_value = f32::NEG_INFINITY;
        let mut result = &compressed_vecs[0];
        for compressed_vec in compressed_vecs {
            let value = self.get_value(compressed_vec.view());
            if (value > max_value) {
                max_value = value;
                result = compressed_vec;
            }
        }
        result
    }

    pub fn get_function_type_id(&self) -> TypeId {
        self.func_type_id
    }

    pub fn get_return_type_id(&self) -> TypeId {
        self.value_field.get_type_id()
    }

    fn get_value(&self, x_compressed : ArrayView1<f32>) -> f32 {
        let func_space_info = self.get_context().get_function_space_info(self.get_function_type_id());
        let y_compressed = func_space_info.apply(self.func_mat.view(), x_compressed);

        self.value_field.get_value_for_compressed_vector(y_compressed.view())
    }

    pub fn get_value_gradient(&self, x_compressed : ArrayView1<f32>) -> Array1<f32> {
        let func_space_info = self.get_context().get_function_space_info(self.get_function_type_id());
        let y_compressed = func_space_info.apply(self.func_mat.view(), x_compressed);

        let func_jacobian = func_space_info.jacobian(self.func_mat.view(), x_compressed);
        let out_feat_jacobian = func_space_info.out_feat_info.get_feature_jacobian(y_compressed.view());

        let feat_vec_coefs = self.value_field.get_feat_vec_coefs();
        let value_field_grad = feat_vec_coefs.dot(&out_feat_jacobian).dot(&func_jacobian);

        let compressed_inv_schmear = &self.value_field.compressed_prior_inv_schmear;
        //The extra constant doesn't matter for gradient computation
        let inv_schmear_part = &compressed_inv_schmear.inv_schmear;
        let diff = &y_compressed - &inv_schmear_part.mean;
        let precision_func_jacobian = inv_schmear_part.precision.dot(&func_jacobian);
        let delta = diff.dot(&precision_func_jacobian);
        let sq_mahalanobis_grad = 2.0f32 * &delta;

        let mut result = value_field_grad;
        result -= &sq_mahalanobis_grad;
        result
    }
}

impl<'a> ArgminOp for ValueFieldMaximumSolver<'a> {
    type Param = Array1<f32>;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, x_compressed : &Self::Param) -> Result<Self::Output, Error> {
        let neg_result = self.get_value(x_compressed.view());
        let result = -neg_result;
        if (result.is_finite()) {
            Ok(result)
        } else {
            Err(anyhow!("Non-finite value for vector: {}", x_compressed))
        }
    }

    fn gradient(&self, x_compressed : &Self::Param) -> Result<Self::Param, Error> {
        let mut neg_result = self.get_value_gradient(x_compressed.view());
        neg_result *= -1.0f32;
        if (all_finite(neg_result.view())) {
            Ok(neg_result)
        } else {
            Err(anyhow!("Non-finite gradient {} for vector {}", neg_result, x_compressed))
        }
    }
}
