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
use ndarray_linalg::trace::Trace;

use argmin::prelude::*;

pub struct LocalFeaturizationInverseSolver {
    pub space_info : Rc<SpaceInfo>,
    pub ellipsoid : Ellipsoid
}

impl LocalFeaturizationInverseSolver {
    pub fn new(space_info : &Rc<SpaceInfo>, ellipsoid : &Ellipsoid,
               init_point : &Array1<f32>) -> LocalFeaturizationInverseSolver {
        let y = space_info.get_features(init_point);
        let diff = y - ellipsoid.center();
        let diff_norm_sq = diff.dot(&diff);
        let diff_norm = diff_norm_sq.sqrt();

        let space_info = Rc::clone(space_info);
        let skew = ellipsoid.skew(); 
        let skew_trace = skew.trace().unwrap();

        if (skew_trace < 0.0f32) {
            error!("Whoa, a negative trace? {}", skew_trace);
            panic!();
        }

        let inv_scale = skew_trace * diff_norm;

        let scale = if (inv_scale == 0.0f32) { 0.0f32} else {100.0f32 / inv_scale};

        let normed_skew = scale * skew;
        let center = ellipsoid.center().clone();
        let ellipsoid = Ellipsoid::new(center, normed_skew);
        LocalFeaturizationInverseSolver {
            space_info,
            ellipsoid
        }
    }
}

impl ArgminOp for LocalFeaturizationInverseSolver {
    type Param = Array1<f32>;
    type Output = f32;
    type Jacobian = ();
    type Hessian = ();
    type Float = f32;

    fn apply(&self, p : &Self::Param) -> Result<Self::Output, Error> {
        let featurized = self.space_info.get_features(p);
        let dist = self.ellipsoid.sq_mahalanobis_dist(&featurized);
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

