use std::hash::*;
use crate::term_application::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use crate::context::*;
use crate::term_input_output::*;

use serde::{Serialize, Deserialize};

///The pairing of a [`TermApplication`] with
///a [`TermReference`] that it evaluated to
///when passed into an [`crate::interpreter_state::InterpreterState`].
#[derive(Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct TermApplicationResult {
    pub term_app : TermApplication,
    pub result_ref : TermReference
}

impl TermApplicationResult {
    pub fn get_term_input_output(&self) -> TermInputOutput {
        TermInputOutput {
            input : self.term_app.arg_ref.clone(),
            output : self.result_ref.clone()
        }
    }
    ///Gets the [`TypeId`] of the argument in the given [`Context`].
    pub fn get_arg_type(&self, ctxt : &Context) -> TypeId {
        self.term_app.get_arg_type(ctxt)
    }
    ///Gets the [`TypeId`] of the function
    pub fn get_func_type(&self) -> TypeId {
        self.term_app.get_func_type()
    }
    ///Gets the [`TypeId`] of the return value in the given [`Context`].
    pub fn get_ret_type(&self, ctxt : &Context) -> TypeId {
        self.term_app.get_ret_type(ctxt)
    }
    ///Gets the argument [`TermReference`] in this [`TermApplicationResult`].
    pub fn get_arg_ref(&self) -> TermReference {
        self.term_app.arg_ref.clone()
    }
    ///Gets the return [`TermReference`] in this [`TermApplicationResult`].
    pub fn get_ret_ref(&self) -> TermReference {
        self.result_ref.clone()
    }
    ///Gets the function [`TermPointer`] in this [`TermApplicationResult`].
    pub fn get_func_ptr(&self) -> TermPointer {
        self.term_app.func_ptr
    }

}
