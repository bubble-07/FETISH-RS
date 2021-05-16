extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::feature_space_info::*;
use crate::data_points::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::data_point::*;

///Represents information that's available about a function type `A -> B`
///in terms of the [`FeatureSpaceInfo`] for the input and output types.
///Here, it is important to bear in mind that the flow of data through
///the application of a typical [`crate::term_model::TermModel`] is:
///`input -(input sketcher)-> compressed input -(input feature mapping)->
/// input features -(model matrix)-> compressed output`.
#[derive(Clone)]
pub struct FunctionSpaceInfo<'a> {
    pub in_feat_info : &'a FeatureSpaceInfo,
    pub out_feat_info : &'a FeatureSpaceInfo
}

impl <'a> FunctionSpaceInfo<'a> {
    ///Gets the number of dimensions for the input feature space.
    pub fn get_feature_dimensions(&self) -> usize {
        self.in_feat_info.feature_dimensions
    }
    ///Gets the output dimensionality of mappings defined by matrices
    ///relative to this [`FunctionSpaceInfo`], so the dimension of the compressed
    ///output space.
    pub fn get_output_dimensions(&self) -> usize {
        self.out_feat_info.get_sketched_dimensions()
    }
    ///Gets the total number of dimensions required to define a model
    ///matrix for a function within this [`FunctionSpaceInfo`].
    pub fn get_full_dimensions(&self) -> usize {
        self.get_feature_dimensions() * self.get_output_dimensions()
    }

    ///Gets the Jacobian for the composite mapping
    ///`compressed input -(input feature mapping)-> input features -(mat)-> compressed output`
    ///evaluated at the given compressed input vector.
    pub fn jacobian(&self, mat : ArrayView2<f32>, input : ArrayView1<f32>) -> Array2<f32> {
        let feat_jacobian = self.in_feat_info.get_feature_jacobian(input);
        let result = mat.dot(&feat_jacobian);
        result
    }
    ///Given a model matrix for a function with this [`FunctionSpaceInfo`] and a compressed
    ///input vector, computes the compressed vector output which results from
    ///applying the function to the argument.
    pub fn apply(&self, mat : ArrayView2<f32>, input : ArrayView1<f32>) -> Array1<f32> {
        let features = self.in_feat_info.get_features(input);
        let result = mat.dot(&features);
        result
    }
    ///Given a [`DataPoints`] whose input/output pairs are both in the input/output compressed
    ///spaces, yields a new [`DataPoints`] whose inputs have been featurized.
    pub fn get_data_points(&self, in_data_points : DataPoints) -> DataPoints {
        let feat_vecs = self.in_feat_info.get_features_mat(in_data_points.in_vecs.view());
        DataPoints {
            in_vecs : feat_vecs,
            ..in_data_points
        }
    }
    ///Given a [`DataPoint`] whose input/output pair are both in the input/output compressed
    ///spaces, yields a new [`DataPoint`] whose input has been featurized.
    pub fn get_data(&self, in_data : DataPoint) -> DataPoint {
        let feat_vec = self.in_feat_info.get_features(in_data.in_vec.view());

        DataPoint {
            in_vec : feat_vec,
            ..in_data
        }
    }
    ///Given a model [`FuncSchmear`] for this [`FunctionSpaceInfo`], and a
    ///[`Schmear`] over compressed inputs, yields an estimated [`Schmear`]
    ///over the result of applying drawn models to drawn inputs.
    pub fn apply_schmears(&self, f : &FuncSchmear, x : &Schmear) -> Schmear {
        let feat_schmear = self.in_feat_info.featurize_schmear(x);
        let result = f.apply(&feat_schmear);
        result
    }
}
