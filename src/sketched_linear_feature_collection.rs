extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

use crate::feature_collection::*;
use crate::params::*;

pub struct SketchedLinearFeatureCollection {
    in_dimensions : usize,
    out_dimensions : usize,
    reg_strength : f32,
    projection_mat : Array2<f32>
}

impl SketchedLinearFeatureCollection {
    pub fn new(in_dimensions : usize) -> SketchedLinearFeatureCollection {
        let out_dimensions = num_sketched_linear_features(in_dimensions);
        let reg_strength = LIN_REG_STRENGTH;
        let projection_mat = Array::random((out_dimensions, in_dimensions), StandardNormal);
        SketchedLinearFeatureCollection {
            in_dimensions,
            out_dimensions,
            reg_strength,
            projection_mat
        }
    }
}

impl FeatureCollection for SketchedLinearFeatureCollection {
    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }

    fn get_dimension(&self) -> usize {
        self.out_dimensions + 1
    }

    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let projected : Array1<f32> = einsum("ab,b->a", &[&self.projection_mat, in_vec]).unwrap()
                                      .into_dimensionality::<Ix1>().unwrap();
        let single_ones = Array::ones((1,));
        stack(Axis(0), &[projected.view(), single_ones.view()]).unwrap()
    }

    fn get_jacobian(&self, in_vec : &Array1<f32>) -> Array2<f32> {
        //The jacobian is given by the projection mat plus the part about cconcatenating
        //the constant 1 [derivative zero w.r.t all vars]
        let zero_row = Array::zeros((1,self.in_dimensions));
        stack(Axis(0), &[self.projection_mat.view(), zero_row.view()]).unwrap()
    }
    fn get_regularization_strength(&self) -> f32 {
        self.reg_strength
    }

}
