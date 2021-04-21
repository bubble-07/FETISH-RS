use fetish_lib::everything::*;
use crate::alpha_formulas::*;
use crate::feature_collection::*;
use crate::params::*;

pub fn build_function_feature_space(arg_space : &FeatureSpaceInfo, 
                                    ret_space : &FeatureSpaceInfo) -> FeatureSpaceInfo {
    let base_dimensions = arg_space.feature_dimensions * ret_space.get_sketched_dimensions();
    build_compressed_feature_space(base_dimensions)
}

pub fn build_compressed_feature_space(base_dimensions : usize) -> FeatureSpaceInfo {
    let reduced_dimensions = get_reduced_dimension(base_dimensions);
    let alpha = sketch_alpha(base_dimensions);
    let sketcher = Option::Some(LinearSketch::new(base_dimensions, reduced_dimensions, alpha));

    let feature_collections = get_default_feature_collections(reduced_dimensions);
    let feature_dimensions = get_total_feat_dims(&feature_collections);
    FeatureSpaceInfo {
        base_dimensions,
        feature_dimensions,
        feature_collections,
        sketcher
    }
}

pub fn build_uncompressed_feature_space(base_dimensions : usize) -> FeatureSpaceInfo {
    let feature_collections = get_default_feature_collections(base_dimensions);
    let feature_dimensions = get_total_feat_dims(&feature_collections);
    FeatureSpaceInfo {
        base_dimensions,
        feature_dimensions,
        feature_collections,
        sketcher : Option::None
    }
}
