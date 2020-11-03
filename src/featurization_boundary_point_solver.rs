extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;
use crate::space_info::*;
use std::rc::*;
use crate::ellipsoid::*;

use argmin::prelude::*;

pub struct FeaturizationBoundaryPointSolver {
    pub space_info : Rc<SpaceInfo>,
    pub ellipsoid : Ellipsoid,
    pub base_point : Array1<f32>,
    pub direction : Array1<f32>
}

impl ArgminOp for FeaturizationBoundaryPointSolver {
    type Param = f32;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, p : &Self::Param) -> Result<Self::Output, Error> {
        let mut x = self.direction.clone();
        x *= *p;
        x += &self.base_point;

        let y = self.space_info.get_features(&x);
        let d = self.ellipsoid.mahalanobis_dist(&y);

        Ok(d - 1.0f32)
    }
}
