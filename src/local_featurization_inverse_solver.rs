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

pub struct LocalFeaturizationInverseSolver {
    pub space_info : Rc<SpaceInfo>,
    pub ellipsoid : Ellipsoid
}

impl ArgminOp for LocalFeaturizationInverseSolver {
    type Param = Array1<f32>;
    type Output = f32;
    type Jacobian = ();
    type Hessian = ();
    type Float = f32;

    fn apply(&self, p : &Self::Param) -> Result<Self::Output, Error> {
        let featurized = self.space_info.get_features(p);
        let dist = self.ellipsoid.mahalanobis_dist(&featurized);
        Ok(dist)
    }
    
    fn gradient(&self, p : &Self::Param) -> Result<Self::Param, Error> {
        let J = self.space_info.get_feature_jacobian(p);
        let y = self.space_info.get_features(p);
        let diff = y - self.ellipsoid.center();
        let p_y = self.ellipsoid.skew();
        let result = J.t().dot(p_y).dot(&diff);
        Ok(result)
    }
}

