use crate::holed_linear_expression::*;
use crate::ellipsoid::*;

pub struct BoundedHoledLinearExpression {
    pub expr : HoledLinearExpression,
    pub bound : Ellipsoid
}
