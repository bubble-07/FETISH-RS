extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;
use crate::count_sketch::*;

use std::sync::Arc;
use rustfft::FFTplanner;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

const QUAD_REG_STRENGTH : f32 = 5.0;
const QUAD_FEATURE_MULTIPLIER : usize = 5;

pub struct QuadraticFeatureCollection {
    in_dimensions : usize,
    reg_strength : f32,
    sketch_one : CountSketch,
    sketch_two : CountSketch,
    fft : Arc<dyn FFT<f32>>,
    ifft : Arc<dyn FFT<f32>>
}

impl QuadraticFeatureCollection {
    pub fn new(in_dimensions : usize) -> QuadraticFeatureCollection {
        let out_dimensions = in_dimensions * QUAD_FEATURE_MULTIPLIER;
        let reg_strength = QUAD_REG_STRENGTH;
        let sketch_one = CountSketch::new(in_dimensions, out_dimensions);
        let sketch_two = CountSketch::new(in_dimensions, out_dimensions);
        
        let mut fftplanner = FFTplanner::<f32>::new(false);
        let mut ifftplanner = FFTplanner::<f32>::new(true);

        let fft = fftplanner.plan_fft(out_dimensions);
        let ifft = ifftplanner.plan_fft(out_dimensions);

        QuadraticFeatureCollection {
            in_dimensions,
            reg_strength,
            sketch_one,
            sketch_two,
            fft,
            ifft
        }
    }
}

fn to_complex(real : f32) -> Complex<f32> {
    Complex::<f32>::new(real, 0.0)
}

fn from_complex(complex : Complex<f32>) -> f32 {
    complex.re
}

impl FeatureCollection for QuadraticFeatureCollection {

    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
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
        result
    }

    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
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
        
        Array::from(first_fft).mapv(from_complex)
    }

    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }

    fn get_dimension(&self) -> usize {
        self.sketch_one.get_out_dimensions()
    }

    fn get_regularization_strength(&self) -> f32 {
        self.reg_strength
    }

}
