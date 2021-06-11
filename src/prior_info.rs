use crate::prior_specification::*;

pub struct PriorInfo {
    pub model_prior_specification : Box<dyn PriorSpecification>,
    pub elaborator_prior_specification : Box<dyn PriorSpecification>
}
