extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use crate::params::*;
use crate::space_info::*;
use crate::type_id::*;
use crate::schmeared_hole::*;
use crate::normal_inverse_wishart::*;
use crate::term_application::*;
use crate::term_reference::*;
use crate::func_schmear::*;
use crate::prior_specification::*;
use crate::func_inverse_schmear::*;
use crate::input_to_schmeared_output::*;
use crate::context::*;

use rand::prelude::*;

use std::collections::HashMap;
use crate::model::*;

///A [`Model`] for a term, with information about what
///prior updates and data updates have been applied as part of the operation
///of the Bayesian embedding process in an [`EmbedderState`].
#[derive(Clone)]
pub struct TermModel<'a> {
    pub type_id : TypeId,
    pub model : Model<'a>,
    prior_updates : HashMap::<TermApplication, NormalInverseWishart>,
    pub data_updates : HashMap::<TermReference, InputToSchmearedOutput>
}

impl <'a> TermModel<'a> {
    ///Gets the [`TypeId`] of the term that this [`TermModel`] is responsible
    ///for learning regression information about.
    pub fn get_type_id(&self) -> TypeId {
        self.type_id
    }
    ///See [`Model::get_total_dims`].
    pub fn get_total_dims(&self) -> usize {
        self.model.get_total_dims()
    }
    ///See [`Model::sample`].
    pub fn sample(&self, rng : &mut ThreadRng) -> Array2<f32> {
        self.model.sample(rng)
    }
    ///See [`Model::sample_as_vec`].
    pub fn sample_as_vec(&self, rng : &mut ThreadRng) -> Array1::<f32> {
        self.model.sample_as_vec(rng)
    }
    ///See [`Model::get_mean_as_vec`].
    pub fn get_mean_as_vec(&self) -> ArrayView1::<f32> {
        self.model.get_mean_as_vec()
    }

    ///See [`Model::get_inverse_schmear`].
    pub fn get_inverse_schmear(&self) -> FuncInverseSchmear {
        self.model.get_inverse_schmear()
    }

    ///See [`Model::get_schmear`].
    pub fn get_schmear(&self) -> FuncSchmear {
        self.model.get_schmear()
    }

    ///Gets the [`SchmearedHole`] in the base space of the type for this [`TermModel`].
    pub fn get_schmeared_hole(&self) -> SchmearedHole {
        let func_type_id = self.get_type_id();
        let inv_schmear =  self.get_inverse_schmear().flatten();

        let result = SchmearedHole {
            type_id : func_type_id,
            inv_schmear
        };
        result
    }

    pub(crate) fn has_data(&self, update_key : &TermReference) -> bool {
        self.data_updates.contains_key(update_key)
    }
    pub(crate) fn update_data(&mut self, update_key : TermReference, data_update : InputToSchmearedOutput) {
        let func_space_info = self.model.ctxt.get_function_space_info(self.get_type_id());
        let feat_update = data_update.featurize(&func_space_info);
        self.model.data += &feat_update;
        self.data_updates.insert(update_key, feat_update);
    }
    pub(crate) fn downdate_data(&mut self, update_key : &TermReference) {
        let data_update = self.data_updates.remove(update_key).unwrap();
        self.model.data -= &data_update;
    }    
    pub(crate) fn has_prior(&self, update_key : &TermApplication) -> bool {
        self.prior_updates.contains_key(update_key)
    }
    pub(crate) fn update_prior(&mut self, update_key : TermApplication, distr : NormalInverseWishart) {
        self.model += &distr;
        self.prior_updates.insert(update_key, distr);
    }
    pub(crate) fn downdate_prior(&mut self, key : &TermApplication) {
        let distr = self.prior_updates.remove(key).unwrap();
        self.model -= &distr;
    }

    ///Constructs a new [`TermModel`] for the given type with the given [`PriorSpecification`]
    ///within the given [`Context`].
    pub fn new(type_id : TypeId, prior_specification : &dyn PriorSpecification,
                                 ctxt : &'a Context) -> TermModel<'a> {
        let prior_updates : HashMap::<TermApplication, NormalInverseWishart> = HashMap::new();
        let data_updates : HashMap::<TermReference, InputToSchmearedOutput> = HashMap::new();
        let arg_type_id = ctxt.get_arg_type_id(type_id);
        let ret_type_id = ctxt.get_ret_type_id(type_id);

        let model = Model::new(prior_specification, arg_type_id, ret_type_id, ctxt);
        TermModel {
            type_id,
            model : model,
            prior_updates : prior_updates,
            data_updates : data_updates
        }
    }
}
