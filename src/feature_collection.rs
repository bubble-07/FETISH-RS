use fetish_lib::everything::*;
use crate::params::*;
use crate::alpha_formulas::*;
use ndarray::*;
use rand::prelude::*;

fn gen_cauchy_random(rng : &mut ThreadRng, dims : usize) -> Array1<f32> {
    generate_cauchy_random(rng, CAUCHY_SCALING, dims)
}

pub fn get_default_feature_collections(in_dimensions : usize) -> Vec<Box<dyn FeatureCollection>> {
    let quadratic_feats = num_quadratic_features(in_dimensions);
    let fourier_feats = num_fourier_features(in_dimensions);
    let linear_feats = num_sketched_linear_features(in_dimensions);

    let linear_alpha = linear_sketched_alpha(in_dimensions, linear_feats);
    let fourier_alpha = fourier_sketched_alpha(fourier_feats);
    let quadratic_alpha = quadratic_sketched_alpha(in_dimensions);

    let fourier_generator = gen_cauchy_random;

    let linear_collection = SketchedLinearFeatureCollection::new(in_dimensions, linear_feats, linear_alpha);
    let quadratic_collection = QuadraticFeatureCollection::new(in_dimensions, quadratic_feats, quadratic_alpha);
    let fourier_collection = FourierFeatureCollection::new(in_dimensions, fourier_feats, 
                                                           fourier_alpha, fourier_generator);

    let mut result = Vec::<Box<dyn FeatureCollection>>::new();
    result.push(Box::new(linear_collection));
    result.push(Box::new(quadratic_collection));
    result.push(Box::new(fourier_collection));

    result
}

