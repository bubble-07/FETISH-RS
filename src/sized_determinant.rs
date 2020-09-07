extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_linalg::solveh::DeterminantH;

#[derive(Clone)]
pub struct SizedDeterminant {
    ln_det_avg : f32,
    size : usize
}

impl SizedDeterminant {
    pub fn eye(size : usize) -> SizedDeterminant {
        SizedDeterminant {
            ln_det_avg : 0.0f32,
            size
        }
    }

    pub fn scale(&mut self, scale : f32) {
        self.ln_det_avg += scale.ln();
    }

    pub fn invert(&mut self) {
        self.ln_det_avg *= -1.0f32;
    }

    pub fn from_psd_matrix(mat : &Array2<f32>) -> SizedDeterminant {
        let (sign, ln_det) = mat.sln_deth().unwrap();
        let size = mat.shape()[0];
        let scale = 1.0f32 / (size as f32);

        let ln_det_avg = ln_det * scale;
        SizedDeterminant {
            ln_det_avg,
            size
        }
    }


    pub fn tensor(&self, other : &SizedDeterminant) -> SizedDeterminant {
        let ret_ln_det_avg = self.ln_det_avg + other.ln_det_avg;
        let ret_size = self.size * other.size;

        SizedDeterminant {
            ln_det_avg : ret_ln_det_avg,
            size : ret_size
        }
    }

    pub fn get_singular_value_geom_mean(&self) -> f32 {
        self.ln_det_avg.exp()
    }
}
