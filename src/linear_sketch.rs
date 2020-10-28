extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use crate::schmear::*;

use crate::inverse_schmear::*;
use crate::params::*;
use crate::pseudoinverse::*;
use crate::test_utils::*;

#[derive(Clone)]
pub struct LinearSketch {
    projection_mat : Array2<f32>,
    projection_mat_pinv : Array2<f32>
}

impl LinearSketch {
    pub fn new(in_dimensions : usize, out_dimensions : usize, alpha : f32) -> LinearSketch {
        let mut projection_mat = Array::random((out_dimensions, in_dimensions), StandardNormal);
        let mut projection_mat_pinv = pseudoinverse(&projection_mat);
        
        projection_mat *= alpha;
        projection_mat_pinv *= (1.0f32 / alpha);

        LinearSketch {
            projection_mat,
            projection_mat_pinv
        }
    }
    pub fn trivial_sketch(dimensions : usize) -> LinearSketch {
        let ident = Array::eye(dimensions);
        LinearSketch {
            projection_mat : ident.clone(),
            projection_mat_pinv : ident
        }
    }

    pub fn compress_inverse_schmear(&self, inv_schmear : &InverseSchmear) -> InverseSchmear {
        inv_schmear.transform_compress(&self.projection_mat)
    }
    pub fn compress_schmear(&self, schmear : &Schmear) -> Schmear {
        schmear.transform_compress(&self.projection_mat)
    }

    pub fn sketch(&self, vec : &Array1<f32>) -> Array1<f32> {
        self.projection_mat.dot(vec)
    }
    pub fn expand(&self, mean : &Array1<f32>) -> Array1<f32> {
        self.projection_mat_pinv.dot(mean)
    }
    pub fn expand_covariance(&self, covariance : &Array2<f32>) -> Array2<f32> {
        self.projection_mat_pinv.dot(covariance).dot(&self.projection_mat_pinv.t())
    }

    pub fn get_expansion_matrix(&self) -> Array2<f32> {
        self.projection_mat_pinv.clone()
    }
    pub fn get_output_dimension(&self) -> usize {
        self.projection_mat.shape()[0]
    }
    pub fn get_input_dimension(&self) -> usize {
        self.projection_mat.shape()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compesss_schmear_and_compress_inv_schmear_align() {
        let linear_sketch = LinearSketch::new(10, 5, 1.0f32);
        let schmear = random_schmear(10);
        let inv_schmear = schmear.inverse();
        let compressed_schmear = linear_sketch.compress_schmear(&schmear);
        let compressed_inv_schmear = linear_sketch.compress_inverse_schmear(&inv_schmear);
        let compressed_schmear_inv = compressed_schmear.inverse();
        assert_equal_inv_schmears(&compressed_inv_schmear, &compressed_schmear_inv);
    }

    #[test]
    fn expand_then_sketch_is_identity() {
        let linear_sketch = LinearSketch::new(20, 10, 1.0f32);
        let vector = random_vector(10); 
        let expanded = linear_sketch.expand(&vector);
        let sketched = linear_sketch.sketch(&expanded);
        assert_equal_vectors(&sketched, &vector);
    }
}
