use crate::ellipsoid::*;
use crate::holed_application::*;
use crate::bounded_holed_linear_expression::*;

pub struct BoundedHoledApplication {
    pub holed_application : HoledApplication,
    pub bound : Ellipsoid
}

impl BoundedHoledApplication {
    pub fn new(holed_application : HoledApplication, bound : Ellipsoid) -> BoundedHoledApplication {
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
