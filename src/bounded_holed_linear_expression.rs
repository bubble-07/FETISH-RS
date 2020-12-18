use crate::holed_linear_expression::*;
use crate::ellipsoid::*;
use crate::bounded_hole::*;
use crate::bounded_holed_application::*;

pub struct BoundedHoledLinearExpression {
    pub expr : HoledLinearExpression,
    pub bound : BoundedHole
}

impl BoundedHoledLinearExpression {
    pub fn get_bounded_hole(&self) -> &BoundedHole {
        &self.bound 
    }
    pub fn extend_with_holed(&self, filler : BoundedHoledApplication) -> BoundedHoledLinearExpression {
        let bound = filler.bound;
        let holed_application = filler.holed_application;
        let extended_expr = self.expr.extend_with_holed(holed_application);
        BoundedHoledLinearExpression {
            bound : bound,
            expr : extended_expr
        }
    }
}
