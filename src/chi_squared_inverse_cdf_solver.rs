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
use statrs::distribution::*;

pub struct ChiSquaredInverseCdfSolver {
    pub distr : ChiSquared, 
    pub quantile : f64
}

impl ArgminOp for ChiSquaredInverseCdfSolver {
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p : &Self::Param) -> Result<Self::Output, Error> {
        let cdf = self.distr.cdf(*p);
        let diff = cdf - self.quantile;
        Result::Ok(diff)
    }
}
