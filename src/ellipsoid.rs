extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;

#[derive(Clone)]
pub struct Ellipsoid {
    inv_schmear : InverseSchmear
}

impl Ellipsoid {
    pub fn contains(&self, vec : &Array1<f32>) -> bool {
        self.mahalanobis_dist(vec) < 1.0f32
    }

    pub fn mahalanobis_dist(&self, vec : &Array1<f32>) -> f32 {
        self.inv_schmear.mahalanobis_dist(vec)
    }
    pub fn transform_compress(&self, mat : &Array2<f32>) -> Ellipsoid {
        let new_inv_schmear = self.inv_schmear.transform_compress(mat);
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }
    //If this is y, and mat is M, propagate an ellipse to x in Mx = y
    pub fn backpropagate_through_transform(&self, mat : &Array2<f32>) -> Ellipsoid {
        let u_y = &self.inv_schmear.mean;
        let s_y = &self.inv_schmear.precision;

        let mat_inv = pseudoinverse_h(mat);
        
        let u_x = mat_inv.dot(u_y);
        let s_x = mat.t().dot(s_y).dot(mat);

        let new_inv_schmear = InverseSchmear {
            mean : u_x,
            precision : s_x
        };
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }

    //If this is y, and we're given x, propagate to an ellipse on Vec(M) in Mx = y
    pub fn backpropagate_to_vectorized_transform(&self, x : &Array1<f32>) -> Ellipsoid {
        let u_y = &self.inv_schmear.mean;
        let s_y = &self.inv_schmear.precision;

        let s = x.shape()[0];
        let t = u_y.shape()[0];
        let d = s * t;

        let mut u_M_full = outer(u_y, x);
        u_M_full *= 1.0f32 / (x.dot(x));
        let u_M = u_M_full.into_shape((d,)).unwrap();

        let x_x_t = outer(x, x);
        let s_M = kron(&x_x_t, s_y);

        let new_inv_schmear = InverseSchmear {
            mean : u_M,
            precision : s_M
        };
        Ellipsoid {
            inv_schmear : new_inv_schmear
        }
    }
}
