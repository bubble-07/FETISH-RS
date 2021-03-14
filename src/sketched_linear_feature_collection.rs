extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

use crate::alpha_formulas::*;
use crate::feature_collection::*;
use crate::params::*;

#[derive(Clone)]
pub struct SketchedLinearFeatureCollection {
    in_dimensions : usize,
    out_dimensions : usize,
    alpha : f32,
    projection_mat : Array2<f32>
}

impl SketchedLinearFeatureCollection {
    pub fn new(in_dimensions : usize) -> SketchedLinearFeatureCollection {
        let out_dimensions = num_sketched_linear_features(in_dimensions);
        let projection_mat = Array::random((out_dimensions, in_dimensions), StandardNormal);

        let alpha = linear_sketched_alpha(in_dimensions, out_dimensions);

        SketchedLinearFeatureCollection {
            in_dimensions,
            out_dimensions,
            alpha,
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
        let projected = self.projection_mat.dot(in_vec);
        let single_ones = Array::ones((1,));
        let result = stack(Axis(0), &[projected.view(), single_ones.view()]).unwrap();
        self.alpha * result
    }

    fn get_jacobian(&self, _in_vec : &Array1<f32>) -> Array2<f32> {
        //The jacobian is given by the projection mat plus the part about cconcatenating
        //the constant 1 [derivative zero w.r.t all vars]
        let zero_row = Array::zeros((1,self.in_dimensions));
        let result = stack(Axis(0), &[self.projection_mat.view(), zero_row.view()]).unwrap();
        self.alpha * result
    }

}
