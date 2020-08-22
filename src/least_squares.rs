extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use std::rc::*;
use crate::model::*;
use crate::params::*;
use crate::test_utils::*;
use crate::inverse_schmear::*;
use crate::linalg_utils::*;
use ndarray_linalg::{LeastSquaresSvd};

//Solves Ax = b with minimal norm determined
//by the passed quadratic form
pub fn least_squares(A : &Array2<f32>, b : &Array1<f32>, Q : &Array2<f32>) -> Array1<f32> {
    let L = sqrtm(Q);
    let LA = L.dot(A);
    let Lb = L.dot(b);
    let x = LA.least_squares(&Lb).unwrap().solution;
    x
}