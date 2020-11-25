use crate::holed_application::*;
use crate::type_id::*;
use crate::linear_expression::*;
use crate::term_reference::*;
use std::vec::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

#[derive(Clone)]
pub struct HoledLinearExpression {
    pub chain : Vec<HoledApplication>
}

impl HoledLinearExpression {
    pub fn get_type(&self) -> TypeId {
        self.chain[0].get_type()
    }
    pub fn get_hole_type(&self) -> TypeId {
        self.chain[self.chain.len() - 1].get_hole_type()
    }
    pub fn cap(&self, cap : TermReference) -> LinearExpression {
        LinearExpression {
            chain : self.clone(),
            cap : cap
        }
    }
    pub fn extend_with_holed(&self, filler : HoledApplication) -> HoledLinearExpression {
        let mut ret_chain = self.chain.clone();
        ret_chain.push(filler);
        HoledLinearExpression {
            chain : ret_chain
        }
    }
}

impl DisplayableWithState for HoledLinearExpression {
    fn display(&self, state : &InterpreterState) -> String {
        let mut result = "-".to_owned();
        for holed_app in self.chain.iter().rev() {
            result = holed_app.format_string(state, result);
        }
        result
    }
}
