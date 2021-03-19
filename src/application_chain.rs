use crate::term_reference::*;
use crate::term_application::*;

#[derive(Clone)]
pub struct ApplicationChain {
    pub term_refs : Vec<TermReference>
}

impl ApplicationChain {
    pub fn add_to_chain(&mut self, term_ref : TermReference) {
        self.term_refs.push(term_ref);
    }
    pub fn from_term_application(term_application : TermApplication) -> ApplicationChain {
        let mut term_refs = Vec::new();
        term_refs.push(TermReference::FuncRef(term_application.func_ptr));
        term_refs.push(term_application.arg_ref);

        ApplicationChain {
            term_refs
        }
    }
}
