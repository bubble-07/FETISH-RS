use std::collections::HashMap;
use crate::interpreter_state::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_application::*;
use crate::func_impl::*;
use crate::model::*;
use crate::model_space::*;

pub struct EmbedderState {
    model_spaces : HashMap::<TypeId, ModelSpace>
}

impl EmbedderState {

    pub fn get_embedding(&self, term_ptr : TermPointer) -> &Model {
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

}

