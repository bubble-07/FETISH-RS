extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;
use crate::count_sketch::*;

use std::sync::Arc;
use rustfft::FFTPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traints::Zero;

const QUAD_REG_STRENGTH : f32 = 5.0;
const QUAD_FEATURE_MULTIPLIER : usize = 5;

struct QuadraticFeatureCollection {
    in_dimensions : usize,
    reg_strength : f32,
    sketch_one : CountSketch,
    sketch_two : CountSketch,
    fft : Arc<dyn FFT<T>>,
    ifft : Arc<dyn FFT<T>>
}

impl QuadraticFeatureCollection {
    fn new(in_dimensions : usize) -> QuadraticFeatureCollection {
        let out_dimensions = in_dimensions * QUAD_FEATURE_MULTIPLIER;
        let reg_strength = QUAD_REG_STRENGTH;
        let sketch_one = CountSketch::new(in_dimensions, out_dimensions);
        let sketch_two = CountSketch::new(in_dimensions, out_dimensions);
        
        let mut fftplanner = FFTPlanner(false);
        let mut ifftplanner = FFTPlanner(true);

        let fft = fftplanner.plan_fft(out_dimensions);
        let ifft = fftplanner.plan_fft(out_dimensions);

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

    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let first_sketch = self.sketch_one.sketch(in_vec)
        let second_sketch = self.sketch_two.sketch(in_vec)

        //FFT polynomial multiplication
        let mut complex_first_sketch = in_vec.mapv(to_complex).to_vec();
        let mut complex_second_sketch = in_vec.mapv(to_complex).to_vec();

        let out_dim = self.get_dimension();
        
        let mut first_fft = vec![Complex::zero(); out_dim];
        let mut second_fft = vec![Complex::zero(); out_dim];

        self.fft.process(&mut complex_first_sketch, &mut first_fft);
        self.fft.process(&mut complex_second_sketch, &mut second_fft);

        //Turn second_fft into the multiplied fft in-place
        for i in 0..out_dim {
            let second_fft[i] *= first_fft[i];
        }

        //Turn first_fft into the result inverse-fft
        self.ifft.process(&mut second_fft, &mut first_fft);
        
        return Array::from(first_fft).mapv(from_complex)
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
