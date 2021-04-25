extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use std::ops;

use crate::context::*;
use crate::prior_specification::*;
use crate::type_id::*;
use crate::data_points::*;
use crate::data_point::*;
use crate::feature_collection::*;
use crate::normal_inverse_wishart::*;
use crate::func_schmear::*;
use crate::func_inverse_schmear::*;
use crate::space_info::*;

use rand::prelude::*;

///A representation of the information known via Bayesian regression
///of a nonlinear model from the compressed space of the given argument [`TypeId`]
///to the compressed space of the given return [`TypeId`]. The wrapped
///[`NormalInverseWishart`] linear model is defined on mappings from the 
///feature space of the input to the compressed space of the output,
///and the whole [`Model`] is assumed to exist in the given [`Context`].
#[derive(Clone)]
pub struct Model<'a> {
    pub arg_type_id : TypeId,
    pub ret_type_id : TypeId,
    pub data : NormalInverseWishart,
    pub ctxt : &'a Context
}

///Convenience method to get all ordered features for a given collection of [`FeatureCollection`]s
///and a given input vector.
pub fn to_features(feature_collections : &Vec<Box<dyn FeatureCollection>>, in_vec : ArrayView1<f32>) -> Array1<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_features(in_vec))
                                   .collect::<Vec<_>>();
    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();

    stack(Axis(0), &comp_views).unwrap()
}

///Convenience method to get all features for a given collection of [`FeatureCollection`]s
///and a given input matrix whose rows are vectors to featurize.
pub fn to_features_mat(feature_collections : &Vec<Box<dyn FeatureCollection>>, in_mat : ArrayView2<f32>)
                      -> Array2<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_features_mat(in_mat))
                                   .collect::<Vec<_>>();
    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();
    stack(Axis(1), &comp_views).unwrap()
}

///Convenience method to get the jacobian of the concatenation of the given
///collection of [`FeatureCollection`]s evaluated at the given input vector.
pub fn to_jacobian(feature_collections : &Vec<Box<dyn FeatureCollection>>, in_vec : ArrayView1<f32>) -> Array2<f32> {
    let comps = feature_collections.iter()
                                   .map(|coll| coll.get_jacobian(in_vec))
                                   .collect::<Vec<_>>();

    let comp_views = comps.iter()
                          .map(|comp| ArrayView::from(comp))
                          .collect::<Vec<_>>();

    stack(Axis(0), &comp_views).unwrap()
}

impl <'a> Model<'a> {
    ///Gets the total number of coefficients used to define the mean of this [`Model`]
    pub fn get_total_dims(&self) -> usize {
        self.data.get_total_dims()
    }
}


impl <'a> Model<'a> {
    ///Gets the [`Context`] that this [`Model`] exists within.
    pub fn get_context(&self) -> &'a Context {
        self.ctxt 
    }
    ///Draws a sample of a linear mapping from the feature space of the input
    ///to the compressed space of the output from the distribution defined by this [`Model`].
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.data.sample(rng)
    }
    ///Identical to [`sample`], but the result is flattened.
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        self.data.sample_as_vec(rng)
    }
    ///Gets the mean of the underlying [`NormalInverseWishart`] model from the feature
    ///space of the input space to the compressed space of the output, as a flattened vector.
    pub fn get_mean_as_vec(&self) -> Array1::<f32> {
        self.data.get_mean_as_vec()
    }
    ///Gets the [`FuncInverseSchmear`] for the underlying [`NormalInverseWishart`] model
    ///from the feature space of the input to the compressed space of the output.
    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        self.data.get_inverse_schmear()
    }

    ///Gets the [`FuncSchmear`] for the underlying [`NormalInverseWishart`] model
    ///from the feature space of the input to the compressed space of the output.
    pub fn get_schmear(&self) -> FuncSchmear {
        self.data.get_schmear()
    }
}

impl <'a> ops::AddAssign<DataPoint> for Model<'a> {
    ///Updates this [`Model`] to reflect new regression information from the
    ///given [`DataPoint`], which is assumed to map from the compressed space
    ///of the input to the compressed space of the output.
    fn add_assign(&mut self, other: DataPoint) {
        let func_space_info = self.ctxt.build_function_space_info(self.arg_type_id, self.ret_type_id);
        self.data += &func_space_info.get_data(other);
    }
}

impl <'a> ops::AddAssign<DataPoints> for Model<'a> {
    ///Updates this [`Model`] to reflect new regression information from the
    ///given [`DataPoints`], which is assumed to map from the compressed space
    ///of the input to the compressed space of the output.
    fn add_assign(&mut self, other : DataPoints) {
        let func_space_info = self.ctxt.build_function_space_info(self.arg_type_id, self.ret_type_id);
        self.data.update_datapoints(&func_space_info.get_data_points(other));
    }
}

impl <'a> ops::SubAssign<DataPoint> for Model<'a> {
    ///Undoes the action of the corresponding `add_assign` method.
    fn sub_assign(&mut self, other: DataPoint) {
        let func_space_info = self.ctxt.build_function_space_info(self.arg_type_id, self.ret_type_id);
        self.data -= &func_space_info.get_data(other);
    }
}

impl <'a> ops::AddAssign<&NormalInverseWishart> for Model<'a> {
    ///Updates this [`Model`] to reflect new prior information from the
    ///given [`NormalInverseWishart`], which is assumed to come from another [`Model`]
    ///with the same input and return spaces.
    fn add_assign(&mut self, other : &NormalInverseWishart) {
        self.data += other;
    }
}

impl <'a> ops::SubAssign<&NormalInverseWishart> for Model<'a> {
    ///Undoes the action of the corresponding `add_assign` method.
    fn sub_assign(&mut self, other : &NormalInverseWishart) {
        self.data -= other;
    }
}

impl <'a> Model<'a> {
    ///Constructs a new [`Model`] with the given [`PriorSpecification`] for the
    ///underlying [`NormalInverseWishart`] model, the given argument and return
    ///types, and existing within the given [`Context`].
    pub fn new(prior_spec : &dyn PriorSpecification,
               arg_type_id : TypeId, ret_type_id : TypeId,
               ctxt : &'a Context) -> Model<'a> {

        let func_space_info = ctxt.build_function_space_info(arg_type_id, ret_type_id);
        let data = NormalInverseWishart::from_space_info(prior_spec, &func_space_info);
    
        Model {
            arg_type_id,
            ret_type_id,
            data,
            ctxt
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;
    use crate::linalg_utils::*;
    use crate::test_utils::*;

    fn clone_model(model : &Model) -> Model {
        let mut result = Model::new(model.arg_type_id, model.ret_type_id);
        result.data = model.data.clone();
        result
    }
    
    fn clone_and_perturb_model(model : &Model, epsilon : f32) -> Model {
        let mut result = Model::new(model.arg_type_id, model.ret_type_id);
        result.data = model.data.clone();
        
        let mean = &model.data.mean;
        let t = mean.shape()[0];
        let s = mean.shape()[1];

        let perturbation = epsilon * random_matrix(t, s);

        result.data.mean += &perturbation;

        result.data.recompute_derived();

        result
    }

    #[test]
    fn data_updates_bulk_matches_incremental() {
        let mut bulk_updated = random_model(*UNARY_VEC_FUNC_T);
        let mut incremental_updated = bulk_updated.clone();

        let mut data_point = random_data_point(DIM, DIM);
        data_point.weight = 1.0f32;

        let mut in_vecs = Array::zeros((1, DIM));
        in_vecs.row_mut(0).assign(&data_point.in_vec);
        let mut out_vecs = Array::zeros((1, DIM));
        out_vecs.row_mut(0).assign(&data_point.out_vec);
        let data_points = DataPoints {
            in_vecs,
            out_vecs
        };

        incremental_updated += data_point;

        bulk_updated += data_points;

        assert_equal_distributions_to_within(&incremental_updated.data, &bulk_updated.data, 1.0f32);
    }

    #[test]
    fn data_updates_undo_cleanly() {
        let expected = random_model(*UNARY_VEC_FUNC_T);

        let mut model = expected.clone();
        let data_point = random_data_point(DIM, DIM);

        model += data_point.clone();
        model -= data_point.clone();

        assert_equal_distributions_to_within(&model.data, &expected.data, 1.0f32);
    }

    #[test]
    fn sampling_accurate() {
        let epsilon = 10.0f32;
        let num_samps = 1000;

        let model = random_model(*UNARY_VEC_FUNC_T);

        let model_schmear = model.get_schmear().flatten();

        let model_dims = model_schmear.mean.shape()[0];

        let mut mean = Array::zeros((model_dims,));
        let mut rng = rand::thread_rng();

        let scale_fac = 1.0f32 / (num_samps as f32);

        for _ in 0..num_samps {
            let sample = model.sample_as_vec(&mut rng);

            mean += &sample;
        }

        mean *= scale_fac;

        assert_equal_vectors_to_within(&mean, &model_schmear.mean, epsilon);


        let mut covariance = Array::zeros((model_dims, model_dims));
        for _ in 0..num_samps {
            let sample = model.sample_as_vec(&mut rng);

            let diff = &sample - &model_schmear.mean;
            covariance += &(scale_fac * &outer(&diff, &diff));
        }

        assert_equal_matrices_to_within(&covariance, &model_schmear.covariance, epsilon * (model_dims as f32));
    }

}
