extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use std::collections::HashMap;
use crate::interpreter_state::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use crate::bayes_utils::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;

pub struct EmbedderState {
    model_spaces : HashMap::<TypeId, ModelSpace>
}

impl EmbedderState {

    pub fn get_embedding(&self, term_ptr : &TermPointer) -> &Model {
        let space : &ModelSpace = self.model_spaces.get(&term_ptr.type_id).unwrap();
        space.get_model(term_ptr.index)
    }

    pub fn get_mut_embedding(&mut self, term_ptr : TermPointer) -> &mut Model {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.get_model_mut(term_ptr.index)
    }

    pub fn init_embedding(&mut self, term_ptr : TermPointer) {
        let space : &mut ModelSpace = self.model_spaces.get_mut(&term_ptr.type_id).unwrap();
        space.add_model(term_ptr.index)
    }

    //Given a TermApplicationResult, compute the estimated output from the application
    //and use it to update the model for the result. If an existing update
    //exists for the given application of terms, this will first remove that update
    fn propagate_prior(&mut self, term_app_res : TermApplicationResult) {
        let func_embedding : &Model = self.get_embedding(&term_app_res.get_arg_ptr());
        let arg_embedding : &Model = self.get_embedding(&term_app_res.get_func_ptr());
        
        //Find the func and arg schmears
        let func_schmear : Schmear = func_embedding.get_schmear();
        let arg_schmear : Schmear = arg_embedding.get_schmear();

        //Get the model space for the func type
        let func_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_func_type()).unwrap();
        let ret_space : &ModelSpace = self.model_spaces.get(&term_app_res.get_ret_type()).unwrap();

        let out_schmear : Schmear = func_space.apply_schmears(&func_schmear, &arg_schmear);
        let out_prior : NormalInverseGamma = ret_space.schmear_to_prior(&out_schmear);

        //Actually perform the update
        let ret_embedding : &mut Model = self.get_mut_embedding(term_app_res.get_ret_ptr());
        if (ret_embedding.has_prior(&term_app_res.term_app)) {
            ret_embedding.downdate_prior(&term_app_res.term_app);
        }
        ret_embedding.update_prior(term_app_res.term_app, out_prior);
    }

    //Given a TermApplicationResult, update the model for the function based on the
    //implicitly-defined data-point for the result
    fn propagate_data(&mut self, term_app_res : TermApplicationResult) {
        let arg_ptr : TermPointer = term_app_res.get_arg_ptr();
        let arg_embedding : &Model = self.get_embedding(&arg_ptr);
        let ret_embedding : &Model = self.get_embedding(&term_app_res.get_ret_ptr());

        let arg_mean : Array1::<f32> = arg_embedding.get_mean_as_vec();
        let out_inv_schmear : InverseSchmear = ret_embedding.get_inverse_schmear();
        let data_point = DataPoint {
            in_vec : arg_mean,
            out_vec : out_inv_schmear.mean,
            out_precision : out_inv_schmear.precision
        };

        let func_embedding : &mut Model = self.get_mut_embedding(term_app_res.get_func_ptr());
        if (func_embedding.has_data(&arg_ptr)) {
            func_embedding.downdate_data(&arg_ptr);
        }
        func_embedding.update_data(arg_ptr, data_point);
    }

}

