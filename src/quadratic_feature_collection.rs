extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::feature_collection::*;
use crate::count_sketch::*;

use std::sync::Arc;
use rustfft::FFTplanner;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use crate::params::*;

use serde::{Serialize, Deserialize};

///A feature collection consisting of sketched quadratic features
///utilizing the [`CountSketch`]-and-[`FFT`] technique
///described in (TODO: cite reference from paper here)
#[derive(Clone)]
pub struct QuadraticFeatureCollection {
    in_dimensions : usize,
    alpha : f32,
    sketch_one : CountSketch,
    sketch_two : CountSketch,
    fft : Arc<dyn FFT<f32>>,
    ifft : Arc<dyn FFT<f32>>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SerializableQuadraticFeatureCollection {
    in_dimensions : usize,
    alpha : f32,
    sketch_one : CountSketch,
    sketch_two : CountSketch
}

impl SerializableQuadraticFeatureCollection {
    pub fn new(in_dimensions : usize, out_dimensions : usize, alpha : f32) -> SerializableQuadraticFeatureCollection {
        let sketch_one = CountSketch::new(in_dimensions, out_dimensions);
        let sketch_two = CountSketch::new(in_dimensions, out_dimensions);
        SerializableQuadraticFeatureCollection {
            in_dimensions,
            alpha,
            sketch_one,
            sketch_two
        }
    }
    pub fn get_dimension(&self) -> usize {
        self.sketch_one.get_out_dimensions()
    }
    pub fn deserialize(self) -> QuadraticFeatureCollection {
        let mut fftplanner = FFTplanner::<f32>::new(false);
        let mut ifftplanner = FFTplanner::<f32>::new(true);

        let out_dimensions = self.sketch_one.get_out_dimensions();

        let fft = fftplanner.plan_fft(out_dimensions);
        let ifft = ifftplanner.plan_fft(out_dimensions);

        QuadraticFeatureCollection {
            in_dimensions : self.in_dimensions,
            alpha : self.alpha,
            sketch_one : self.sketch_one,
            sketch_two : self.sketch_two,
            fft,
            ifft
        }
    }
}

impl QuadraticFeatureCollection {
    ///Constructs a new [`QuadraticFeatureCollection`] with the given number of input
    ///dimensions, the given scaling factor `alpha`], and the given number of quadratic
    ///features `out_dimensions`.
    pub fn new(in_dimensions : usize, out_dimensions : usize, alpha : f32) -> QuadraticFeatureCollection {
        let serializable = SerializableQuadraticFeatureCollection::new(in_dimensions, out_dimensions, alpha);
        serializable.deserialize()
    }

    pub fn serialize(self) -> SerializableQuadraticFeatureCollection {
        SerializableQuadraticFeatureCollection {
            in_dimensions : self.in_dimensions,
            alpha : self.alpha,
            sketch_one : self.sketch_one,
            sketch_two : self.sketch_two
        }
    }
}

fn to_complex(real : f32) -> Complex<f32> {
    Complex::<f32>::new(real, 0.0)
}

fn from_complex(complex : Complex<f32>) -> f32 {
    complex.re
}

impl QuadraticFeatureCollection {
    ///Unoptimized implementation of "get_features", for testing purposes
    fn unoptimized_get_features(&self, in_vec : ArrayView1<f32>) -> Array1<f32> {
        let s = self.in_dimensions;
        let t = self.get_dimension();
        
        let mut result : Array1<f32> = Array::zeros((t,));
        for i in 0..s {
            for j in 0..s {
                let x = in_vec[[i,]];
                let y = in_vec[[j,]];
                let sign = self.sketch_one.signs[i] * self.sketch_two.signs[j];
                let index = (self.sketch_one.indices[i] + self.sketch_two.indices[j]) % t;
                result[[index,]] += sign * x * y;
            }
        }
        self.alpha * result
    }
}

impl FeatureCollection for QuadraticFeatureCollection {

    fn get_jacobian(&self, in_vec: ArrayView1<f32>) -> Array2<f32> {
        //Yield the t x s jacobian of the feature mapping
        //since the feature mapping here is a circular convolution
        //of sketched versions of the input features,
        //we will actually wind up computing our output manually
        let s = self.in_dimensions;
        let t = self.get_dimension();

        let mut result : Array2<f32> = Array::zeros((t, s));
        for i in 0..s {
            for j in 0..s {
                let x = in_vec[[i,]];
                let y = in_vec[[j,]];
                let sign = self.sketch_one.signs[i] * self.sketch_two.signs[j];
                let index = (self.sketch_one.indices[i] + self.sketch_two.indices[j]) % t;
                
                result[[index, i]] += sign * y;
                result[[index, j]] += sign * x;
            }
        }
        self.alpha * result
    }

    fn get_features(&self, in_vec: ArrayView1<f32>) -> Array1<f32> {
        let first_sketch = self.sketch_one.sketch(in_vec);
        let second_sketch = self.sketch_two.sketch(in_vec);

        //FFT polynomial multiplication
        let mut complex_first_sketch = first_sketch.mapv(to_complex).to_vec();
        let mut complex_second_sketch = second_sketch.mapv(to_complex).to_vec();

        let out_dim = self.get_dimension();
        
        let mut first_fft = vec![Complex::zero(); out_dim];
        let mut second_fft = vec![Complex::zero(); out_dim];

        self.fft.process(&mut complex_first_sketch, &mut first_fft);
        self.fft.process(&mut complex_second_sketch, &mut second_fft);

        //Turn second_fft into the multiplied fft in-place
        for i in 0..out_dim {
            second_fft[i] *= first_fft[i];
        }

        //Turn first_fft into the result inverse-fft
        self.ifft.process(&mut second_fft, &mut first_fft);

        //Normalize [since the fft library does unnormalized ffts]
        let scale_fac : f32 = 1.0 / (out_dim as f32);
        for i in 0..out_dim {
            first_fft[i] *= scale_fac;
        }
        
        let result = Array::from(first_fft).mapv(from_complex);
        self.alpha * result
    }

    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }

    fn get_dimension(&self) -> usize {
        self.sketch_one.get_out_dimensions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn empirical_jacobian_is_jacobian() {
        let quadratic_feature_collection = QuadraticFeatureCollection::new(10, 15, 1.0f32);
        let in_vec = random_vector(10);
        let jacobian = quadratic_feature_collection.get_jacobian(in_vec.view());
        let empirical_jacobian = empirical_jacobian(|x| quadratic_feature_collection.get_features(x),
                                                        in_vec.view());
        assert_equal_matrices_to_within(jacobian.view(), empirical_jacobian.view(), 0.1f32);
    }

    #[test]
    fn unoptimized_get_features_is_get_features() {
        let quadratic_feature_collection = QuadraticFeatureCollection::new(10, 15, 1.0f32);
        let in_vec = random_vector(10);
        let unoptimized = quadratic_feature_collection.unoptimized_get_features(in_vec.view());
        let optimized = quadratic_feature_collection.get_features(in_vec.view());
        assert_equal_vectors(optimized.view(), unoptimized.view());
    }
}
