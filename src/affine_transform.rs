extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use crate::sqrtm::*;
use crate::ellipsoid_sampler::*;
use crate::array_utils::*;
use crate::pseudoinverse::*;
use crate::linalg_utils::*;
use crate::inverse_schmear::*;
use crate::featurized_points::*;
use crate::space_info::*;
use crate::test_utils::*;
use crate::params::*;
use crate::ellipsoid::*;
use crate::schmear::*;
use crate::rand_utils::*;
use crate::func_ellipsoid::*;
use std::rc::*;

#[derive(Clone)]
pub struct AffineTransform {
    pub offset : Array1<f32>,
    pub shear : Array2<f32>
}

impl AffineTransform {
    pub fn inverse(&self) -> AffineTransform {
        let shear = pseudoinverse(&self.shear);
        let mut offset = shear.dot(&self.offset);
        offset *= -1.0f32;
        AffineTransform {
            offset,
            shear
        }
    }
    pub fn apply(&self, in_vec : &Array1<f32>) -> Array1<f32> {
        let mut result = self.shear.dot(in_vec);
        result += &self.offset;
        result
    }
    pub fn compose(&self, other : &AffineTransform) -> AffineTransform {
        let shear = self.shear.dot(&other.shear);
        let transformed_other_offset = self.shear.dot(&other.offset);
        let offset = transformed_other_offset + &self.offset;
        let result = AffineTransform {
            offset,
            shear
        };
        result
    }
    pub fn transform_schmear(&self, schmear : &Schmear) -> Schmear {
        let mean = self.apply(&schmear.mean);
        let covariance = self.shear.dot(&schmear.covariance).dot(&self.shear.t());
        Schmear {
            mean,
            covariance
        }
    }
    pub fn transform_ellipsoid(&self, ellipsoid : &Ellipsoid) -> Ellipsoid {
        let mean = ellipsoid.center().clone();
        let covariance = pseudoinverse_h(ellipsoid.skew());
        let schmear = Schmear {
            mean,
            covariance
        };
        let transformed_schmear = self.transform_schmear(&schmear);
        let transformed_inv_schmear = transformed_schmear.inverse();
        let result = Ellipsoid::new(transformed_inv_schmear.mean, transformed_inv_schmear.precision);
        result
    }
    pub fn schmear_standardizing_transform(schmear : &Schmear) -> AffineTransform {
        let inv_shear = sqrtm(&schmear.covariance);
        let inv_offset = &schmear.mean;
        let inv_transform = AffineTransform {
            shear : inv_shear,
            offset : inv_offset.clone()
        };
        inv_transform.inverse()
    }
}
