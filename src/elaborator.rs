use ndarray::*;
use ndarray_linalg::*;
use rand::prelude::*;
use crate::type_id::*;
use crate::params::*;
use crate::schmear::*;
use crate::func_schmear::*;
use crate::func_scatter_tensor::*;
use crate::space_info::*;
use crate::normal_inverse_wishart::*;
use crate::data_point::*;
use crate::func_schmear::*;
use crate::sigma_points::*;
use crate::model::*;
use crate::prior_specification::*;
use crate::context::*;
use std::collections::HashMap;
use crate::term_index::*;

//Learned "opposite" of the sketcher for a given type

type ModelKey = TermIndex;

///Learned left-inverse to the sketcher for a given type.
///Given compressed vectors, an [`Elaborator`] represents
///information about likely-to-be-seen vectors
///in the base space which project to it.
pub struct Elaborator<'a> {
    pub type_id : TypeId,
    ///The linear model here is from projected vectors to vectors expressed in terms of the
    ///orthogonal basis for the kernel of the projection for the given type.
    pub model : NormalInverseWishart,
    ///Stored collection of [`DataPoint`] updates that have been applied to this [`Elaborator`]
    ///indexed by the [`TermIndex`]es of terms that they originated from.
    pub updates : HashMap::<ModelKey, Vec<DataPoint>>,
    pub ctxt : &'a Context
}

struct ElaboratorPrior {
}
impl PriorSpecification for ElaboratorPrior {
    fn get_in_precision_multiplier(&self, _feat_dims : usize) -> f32 {
        ELABORATOR_IN_PRECISION_MULTIPLIER
    }
    fn get_out_covariance_multiplier(&self, out_dims : usize) -> f32 { 
        //We'll directly tinker with the mean covariance schmear's size
        let pseudo_observations = self.get_out_pseudo_observations(out_dims);
        pseudo_observations * ELABORATOR_OUT_COVARIANCE_MULTIPLIER
    }
    fn get_out_pseudo_observations(&self, out_dims : usize) -> f32 {
        //The +4 is because we need to ensure that we always have
        //a valid covariance schmear for this model. See Wikipedia
        //page on the Inverse-Wishart distribution's variance
        (out_dims as f32) * ELABORATOR_ERROR_COVARIANCE_PRIOR_OBSERVATIONS_PER_DIMENSION + 4.0f32
    }
}

impl<'a> Elaborator<'a> {
    ///Constructs a new [`Elaborator`] for the given [`TypeId`] in the given [`Context`].
    ///Before calling this, you should make sure that there is in fact a [`LinearSketch`]
    ///for the given type, and that it has a kernel. There's no point in creating one of these
    ///otherwise.
    pub fn new(type_id : TypeId, ctxt : &'a Context) -> Elaborator<'a> {
        let feature_space_info = ctxt.get_feature_space_info(type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let sketched_dimension = sketcher.get_output_dimension();
        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let kernel_basis_dimension = kernel_mat.shape()[1];

        let prior_specification = ElaboratorPrior {};
        let model = NormalInverseWishart::from_in_out_dims(&prior_specification,
                                                           sketched_dimension, kernel_basis_dimension);

        Elaborator {
            type_id,
            model,
            updates : HashMap::new(),
            ctxt
        }
    }

    ///Gets the mean of the distribution that this [`Elaborator`] defines over the left-inverse
    ///of the projection for this elaborator's type.
    pub fn get_mean(&self) -> Array2<f32> {
        let feature_space_info = self.ctxt.get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let expansion_mat = sketcher.get_expansion_matrix();

        let model_sample = &self.model.mean;
        let mut expanded_model_sample = kernel_mat.dot(model_sample);
        expanded_model_sample += expansion_mat;

        expanded_model_sample
    }

    ///Samples a left-inverse to the projection for this elaborator's type from the
    ///distribution defined by this [`Elaborator`].
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        let feature_space_info = self.ctxt.get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let expansion_mat = sketcher.get_expansion_matrix();

        let model_sample = self.model.sample(rng);
        let mut expanded_model_sample = kernel_mat.dot(&model_sample);
        expanded_model_sample += expansion_mat;

        expanded_model_sample
    }

    ///Using the distribution that this [`Elaborator`] defines over possible
    ///expansions of vectors, and given a [`Schmear`] in the compressed space,
    ///yields the [`Schmear`] in the expanded space which corresponds to sampling
    ///possible expansion matrices from the [`Elaborator`] and vectors from the
    ///passed in [`Schmear`], and applying the former to the latter.
    pub fn expand_schmear(&self, compressed_schmear : &Schmear) -> Schmear {
        let expansion_func_schmear = self.get_expansion_func_schmear();
        expansion_func_schmear.apply(compressed_schmear)
    }

    ///Gets the [`FuncSchmear`] that this [`Elaborator`] defines over left-inverses
    ///to the projection matrix for the type of this elaborator.
    pub fn get_expansion_func_schmear(&self) -> FuncSchmear {
        let feature_space_info = self.ctxt.get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let expansion_mat = sketcher.get_expansion_matrix();

        let kernel_mat = sketcher.get_kernel_matrix().as_ref().unwrap();
        let kernel_mat_t_temp = kernel_mat.t();
        let kernel_mat_t = kernel_mat_t_temp.as_standard_layout();

        //dims: sketched_dimension -> kernel_basis_dimension
        let model_func_schmear = self.model.get_schmear();
        let model_mean = &model_func_schmear.mean;
        let model_out_covariance = &model_func_schmear.covariance.out_scatter;

        //We need to compute a func schmear of dims:
        //sketched_dimension -> full_dimension
        //As the sum of the usual pseudoinverse of the projection
        //plus the appropriately-transformed version of self.model's schmear
        let result_mean = expansion_mat + &kernel_mat.dot(model_mean);

        let result_out_covariance = kernel_mat.dot(model_out_covariance).dot(&kernel_mat_t);

        let result_covariance = FuncScatterTensor {
            in_scatter : model_func_schmear.covariance.in_scatter,
            out_scatter : result_out_covariance
        };
        
        let result_schmear = FuncSchmear {
            mean : result_mean,
            covariance : result_covariance
        };
        result_schmear
    }

    ///Returns true if this [`Elaborator`] has an update stemming from the given
    ///[`TermIndex`] that has been applied to it.
    pub fn has_data(&self, update_key : &ModelKey) -> bool {
        self.updates.contains_key(update_key)
    }
    ///Given a [`Model`] for a term with the given [`TermIndex`], updates this
    ///[`Elaborator`] to reflect that the passed [`Model`] should be something
    ///that this [`Elaborator`] does a good job of faithfully representing through
    ///a round-trip projection for the type -> expansion with the [`Elaborator`].
    pub fn update_data(&mut self, update_key : ModelKey, data_update : &Model) {
        let feature_space_info = self.ctxt.get_feature_space_info(self.type_id);
        let sketcher = &feature_space_info.sketcher.as_ref().unwrap();
        let kernel_mat = &sketcher.get_kernel_matrix().as_ref().unwrap();

        let func_mean = data_update.get_mean_as_vec();

        let mut data_updates = Vec::new();

        let sketched = sketcher.sketch(func_mean.view());
        let expanded = sketcher.expand(sketched.view());
        let diff = func_mean - &expanded;
        let diff_in_kernel_basis = kernel_mat.t().dot(&diff);

        let data_update = DataPoint {
            in_vec : sketched,
            out_vec : diff_in_kernel_basis,
            weight : 1.0f32
        };

        self.model += &data_update;

        data_updates.push(data_update);

        self.updates.insert(update_key, data_updates);
    }
    ///Undoes an update added for the given [`TermIndex`] using [`update_data`]
    pub fn downdate_data(&mut self, update_key : &ModelKey) {
        let mut data_updates = self.updates.remove(update_key).unwrap();
        for data_update in data_updates.drain(..) {
            self.model -= &data_update;
        }
    }
}
