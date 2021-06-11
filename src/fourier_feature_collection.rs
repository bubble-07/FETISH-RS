extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::feature_collection::*;
use crate::linalg_utils::*;
use rand::prelude::*;
use crate::params::*;

use serde::{Serialize, Deserialize};

///A feature collection of random fourier features
#[derive(Clone, Serialize, Deserialize)]
pub struct FourierFeatureCollection {
    in_dimensions : usize,
    num_features : usize,
    alpha : f32,
    ws : Array2<f32> //Matrix which is num_features x in_dimensions
}

impl FourierFeatureCollection {
    ///Constructs a new collection of (alpha-scaled) random fourier features for the given number of
    ///input dimensions, the given number of features, and the given closure which samples random
    ///coefficient vectors for the fourier features.
    ///Concretely, the features are of the form `x -> alpha * <w_i, x>`, where each `w_i` is
    ///a sampled coefficient vector.
    pub fn new(in_dimensions: usize, num_features : usize,
               alpha : f32,
               generator : fn(&mut ThreadRng, usize) -> Array1<f32>) -> FourierFeatureCollection {

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
    fn get_features(&self, in_vec: ArrayView1<f32>) -> Array1<f32> {
        let dotted = self.ws.dot(&in_vec);
        let sine = dotted.mapv(f32::sin);
        let cosine = dotted.mapv(f32::cos);
        
        let result = stack(Axis(0), &[sine.view(), cosine.view()]).unwrap();

        self.alpha * result
    }

    fn get_jacobian(&self, in_vec: ArrayView1<f32>) -> Array2<f32> {
        //The derivative is of the form d/dx f(Wx) = J_f(Wx) W x
        //only here, J_f(Wx) is the concatenation of two diagonal mats
        //Get the dotted vector, and compute the components of J_f(Wx)
        let dotted = self.ws.dot(&in_vec);
        let cos = dotted.mapv(f32::cos);
        let neg_sine = -dotted.mapv(f32::sin);
        
        let part_one = scale_rows(self.ws.view(), cos.view());
        let part_two = scale_rows(self.ws.view(), neg_sine.view());
        
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
    use crate::test_utils::*;
    use crate::rand_utils::*;

    #[test]
    fn empirical_jacobian_is_jacobian() {
        let mut successes : usize = 0;
        for _ in 0..10 {
            let fourier_feature_collection = FourierFeatureCollection::new(10, 15, 1.0f32, gen_nsphere_random);
            let in_vec = random_vector(10);
            let jacobian = fourier_feature_collection.get_jacobian(in_vec.view());
            let empirical_jacobian = empirical_jacobian(|x| fourier_feature_collection.get_features(x),
                                                            in_vec.view());
            let test = are_equal_matrices_to_within(jacobian.view(), empirical_jacobian.view(), 1.0f32, false);
            if (test) {
                successes += 1;
            }
        }
        if (successes < 5) {
            panic!();
        }
    }
}
