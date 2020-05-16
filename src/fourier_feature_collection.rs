extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use crate::feature_collection::*;
use rand::prelude::*;

const FOURIER_REG_STRENGTH : f32 = 1000.0;
const FOURIER_FEATURE_MULTIPLIER : usize = 20;

pub struct FourierFeatureCollection {
    in_dimensions : usize,
    reg_strength : f32,
    num_features : usize,
    ws : Array2<f32> //Matrix which is in_dimensions x num_features
}

impl FourierFeatureCollection {
    pub fn new(in_dimensions: usize, generator : fn(&mut ThreadRng, usize) -> Array1<f32>) -> FourierFeatureCollection {
        let reg_strength = FOURIER_REG_STRENGTH;
        let num_features = FOURIER_FEATURE_MULTIPLIER * in_dimensions;

        let mut ws = Array::zeros((in_dimensions, num_features));
        let mut rng = rand::thread_rng();
        for i in 0..num_features {
            let feature = generator(&mut rng, in_dimensions);
            for j in 0..in_dimensions {
                ws[[i,j]] = feature[[j,]];
            }
        }

        FourierFeatureCollection {
            in_dimensions,
            reg_strength,
            num_features,
            ws
        }
    }
}

impl FeatureCollection for FourierFeatureCollection {
    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let dotted = self.ws.dot(in_vec);
        let sine = dotted.mapv(f32::sin);
        let cosine = dotted.mapv(f32::cos);
        
        stack(Axis(0), &[sine.view(), cosine.view()]).unwrap()
    }

    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }
    
    fn get_dimension(&self) -> usize {
        self.num_features * 2
    }

    fn get_regularization_strength(&self) -> f32 {
        self.reg_strength
    }
}
