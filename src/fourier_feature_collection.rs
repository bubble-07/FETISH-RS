extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::alpha_formulas::*;
use crate::feature_collection::*;
use crate::linalg_utils::*;
use rand::prelude::*;
use crate::params::*;
use crate::test_utils::*;
use crate::rand_utils::*;

#[derive(Clone)]
pub struct FourierFeatureCollection {
    in_dimensions : usize,
    num_features : usize,
    alpha : f32,
    ws : Array2<f32> //Matrix which is num_features x in_dimensions
}

impl FourierFeatureCollection {
    pub fn new(in_dimensions: usize, generator : fn(&mut ThreadRng, usize) -> Array1<f32>) -> FourierFeatureCollection {
        let num_features = num_fourier_features(in_dimensions);

        let alpha = fourier_sketched_alpha(num_features);

        let mut ws = Array::zeros((num_features, in_dimensions));
        let mut rng = rand::thread_rng();
        for i in 0..num_features {
            let feature = generator(&mut rng, in_dimensions);
            for j in 0..in_dimensions {
                ws[[i, j]] = feature[[j,]];
            }
        }

        FourierFeatureCollection {
            in_dimensions,
            num_features,
            alpha,
            ws
        }
    }
}

impl FeatureCollection for FourierFeatureCollection {
    fn get_features(&self, in_vec: &Array1<f32>) -> Array1<f32> {
        let dotted = self.ws.dot(in_vec);
        let sine = dotted.mapv(f32::sin);
        let cosine = dotted.mapv(f32::cos);
        
        let result = stack(Axis(0), &[sine.view(), cosine.view()]).unwrap();

        self.alpha * result
    }

    fn get_jacobian(&self, in_vec: &Array1<f32>) -> Array2<f32> {
        //The derivative is of the form d/dx f(Wx) = J_f(Wx) W x
        //only here, J_f(Wx) is the concatenation of two diagonal mats
        //Get the dotted vector, and compute the components of J_f(Wx)
        let dotted = self.ws.dot(in_vec);
        let cos = dotted.mapv(f32::cos);
        let neg_sine = -dotted.mapv(f32::sin);
        
        let part_one = scale_rows(&self.ws, &cos);
        let part_two = scale_rows(&self.ws, &neg_sine);
        
        let result = stack(Axis(0), &[part_one.view(), part_two.view()]).unwrap()
            .into_dimensionality::<Ix2>().unwrap();

        self.alpha * result
    }

    fn get_in_dimensions(&self) -> usize {
        self.in_dimensions
    }
    
    fn get_dimension(&self) -> usize {
        self.num_features * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_jacobian_is_jacobian() {
        let mut successes : usize = 0;
        for i in 0..10 {
            let fourier_feature_collection = FourierFeatureCollection::new(10, gen_cauchy_random);
            let in_vec = random_vector(10);
            let jacobian = fourier_feature_collection.get_jacobian(&in_vec);
            let empirical_jacobian = empirical_jacobian(|x| fourier_feature_collection.get_features(x),
                                                            &in_vec);
            let test = are_equal_matrices_to_within(&jacobian, &empirical_jacobian, 1.0f32, false);
            if (test) {
                successes += 1;
            }
        }
        if (successes < 5) {
            panic!();
        }
    }
}
