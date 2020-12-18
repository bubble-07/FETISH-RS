extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::ellipsoid::*;
use crate::ellipsoid_sampler::*;
use crate::array_utils::*;
use crate::func_scatter_tensor::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;
use crate::featurized_points::*;
use crate::space_info::*;
use crate::func_inverse_schmear::*;
use crate::test_utils::*;
use crate::params::*;
use crate::rand_utils::*;
use crate::local_featurization_inverse_solver::*;
use crate::featurization_boundary_point_solver::*;
use crate::minimum_volume_enclosing_ellipsoid::*;
use std::rc::*;

#[derive(Clone)]
pub struct FuncEllipsoid {
    inv_schmear : FuncInverseSchmear
}

impl FuncEllipsoid {
    pub fn new(center : Array2<f32>, precision : FuncScatterTensor) -> FuncEllipsoid {
        let inv_schmear = FuncInverseSchmear {
            mean : center,
            precision
        };
        FuncEllipsoid {
            inv_schmear
        }
    }
    pub fn flatten(&self) -> Ellipsoid {
        let inv_schmear = self.inv_schmear.flatten();
        let result = Ellipsoid::new(inv_schmear.mean, inv_schmear.precision);
        result
    }
    pub fn compress(&self, mat : &Array2<f32>) -> Ellipsoid {
        let func_schmear = self.inv_schmear.inverse();
        let schmear = func_schmear.compress(mat);
        let inv_schmear = schmear.inverse();
        Ellipsoid::new(inv_schmear.mean, inv_schmear.precision)
    }
}
