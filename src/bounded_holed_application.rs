use crate::ellipsoid::*;
use crate::holed_application::*;
use crate::bounded_holed_linear_expression::*;
use crate::bounded_hole::*;

pub struct BoundedHoledApplication {
    pub holed_application : HoledApplication,
    pub bound : BoundedHole
}

impl BoundedHoledApplication {
    pub fn new(holed_application : HoledApplication, bound : BoundedHole) -> BoundedHoledApplication {
        BoundedHoledApplication {
            holed_application,
            bound
        }
    }
    pub fn to_linear_expression(&self) -> BoundedHoledLinearExpression {
        BoundedHoledLinearExpression {
            expr : self.holed_application.to_linear_expression(),
            bound : self.bound.clone()
        }
    }
}
