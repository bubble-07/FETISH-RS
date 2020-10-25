use crate::holed_application::*;
use crate::type_id::*;
use std::vec::*;

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
}
