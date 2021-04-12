extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::pseudoinverse::*;
use crate::compressed_inv_schmear::*;

///Represents a (multivariate) probability distribution by its
///mean and its precision matrix, which is the inverse of covariance.
///See also [`crate::schmear::Schmear`], [`crate::func_schmear::FuncSchmear`]
#[derive(Clone)]
pub struct InverseSchmear {
    pub mean : Array1<f32>,
    pub precision : Array2<f32>
}

impl InverseSchmear {
    ///Given a matrix M representing a linear transformation _from_ a space
    ///of smaller dimension _to_ the dimension of this [`InverseSchmear`],
    ///yields the [`CompressedInverseSchmear`] representing the quadratic
    ///form `x -> (u - Mx)^T P (u - Mx)`, where `u` and `P` are this [`InverseSchmear`]'s
    ///mean vector and precision matrix, respectively
    pub fn compress(&self, expansion_mat : ArrayView2<f32>) -> CompressedInverseSchmear {
        let m_t_lambda = expansion_mat.t().dot(&self.precision);
        let Q = m_t_lambda.dot(&expansion_mat);
        let z = m_t_lambda.dot(&self.mean);

        let u_t_lambda_u = self.mean.dot(&self.precision).dot(&self.mean);
        let z_t_Q_z = z.dot(&Q).dot(&z);
        let extra_sq_distance = u_t_lambda_u - z_t_Q_z;

        let Q_inv = pseudoinverse_h(&Q);

        let mean = Q_inv.dot(&z);
        let precision = Q;
        
        let inv_schmear = InverseSchmear {
            mean,
            precision
        };

        let result = CompressedInverseSchmear {
            inv_schmear,
            extra_sq_distance
        };
        result
    }

    ///Scales the precision matrix by the passed scale factor
    pub fn rescale_spread(&self, scale_fac : f32) -> InverseSchmear {
        let mut precision = self.precision.clone();
        precision *= scale_fac;

        InverseSchmear {
            mean : self.mean.clone(),
            precision
        }
    }
    ///Computes the squared Mahalanobis distance
    ///`(u - x)^T P (u - x)` of the argument, where `u` and `P` are this [`InverseSchmear`]'s
    ///mean vector and precision matrix, respectively, and `x` is the passed vector.
    pub fn sq_mahalanobis_dist(&self, vec : ArrayView1<f32>) -> f32 {
        let diff = &vec - &self.mean;
        let precision_diff = self.precision.dot(&diff);
        let result : f32 = diff.dot(&precision_diff);
        result
    }

    ///Creates a [`InverseSchmear`] with the given mean and an all-zero precision matrix.
    pub fn zero_precision_from_vec(vec : Array1<f32>) -> InverseSchmear {
        let n = vec.len();
        let precision = Array::zeros((n, n));
        InverseSchmear {
            mean : vec,
            precision
        }
     }
}
