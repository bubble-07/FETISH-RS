use crate::holed_application::*;
use crate::holed_linear_expression::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;

pub struct LinearExpression {
    pub chain : HoledLinearExpression,
    pub cap : TermReference
}

impl LinearExpression {
    pub fn get_type(&self) -> TypeId {
        self.chain.get_type()
    }
}
