extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use crate::array_utils::*;
use ndarray_linalg::*;
use noisy_float::prelude::*;
use crate::linalg_utils::*;

use crate::inverse_schmear::*;
use crate::pseudoinverse::*;

///Represents a (multivariate) probability distribution by its
///mean and its covariance.
///See also [`InverseSchmear`], [`crate::func_schmear::FuncSchmear`]
#[derive(Clone)]
pub struct Schmear {
    pub mean : Array1<f32>,
    pub covariance : Array2<f32>
}

const LN_TWO_PI : f32 = 1.83787706641f32;

impl Schmear {
    ///Given a collection of points representing
    ///samples from some distribution, returns the empirical
    ///[`Schmear`] describing the sample distribution.
    pub fn from_sample_vectors(vecs : &Vec<Array1<f32>>) -> Schmear {
        let d = vecs[0].shape()[0];
        let n = vecs.len();
        let one_over_n = (1.0f32 / (n as f32));
        let one_over_n_minus_one = (1.0f32 / ((n - 1) as f32));

        let mut mean = Array::zeros((d,));
        for vec in vecs.iter() {
            mean += vec;
        }
        mean *= one_over_n;

        let mut covariance = Array::zeros((d, d));
        for vec in vecs.iter() {
            covariance += &outer(vec, vec);
        }
        covariance *= one_over_n_minus_one;
        Schmear {
            mean,
            covariance
        }
    }
    
    ///Constructs a [`Schmear`] whose mean is the passed vector,
    ///and covariance the zero matrix.
    pub fn from_vector(vec : &Array1<R32>) -> Schmear {
        let n = vec.len();
        let mean = from_noisy(vec);
        let covariance : Array2::<f32> = Array::zeros((n, n));
        Schmear {
            mean : mean,
            covariance : covariance
        }
    }
    ///Constructs the [`InverseSchmear`] which corresponds to this
    ///[`Schmear`] by taking the [`pseudoinverse_h`] of the covariance.
    pub fn inverse(&self) -> InverseSchmear {
        let mean = self.mean.clone();
        let precision = pseudoinverse_h(&self.covariance);
        InverseSchmear {
            mean,
            precision
        }
    }
    ///Given a matrix representing a coordinate-system transformation,
    ///yields this [`Schmear`] under the image of the linear transform.
    pub fn transform(&self, mat : &Array2<f32>) -> Schmear {
        let mean = mat.dot(&self.mean);
        let covariance = mat.dot(&self.covariance).dot(&mat.t());
        Schmear {
            mean,
            covariance
        }
    }
}
