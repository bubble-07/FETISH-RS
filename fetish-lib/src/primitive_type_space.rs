use crate::type_id::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::func_impl::*;
use rand::Rng;
use std::collections::HashMap;

///Collection of primitive function terms, which are just
///[`FuncImpl`]s, for the given type.
pub struct PrimitiveTypeSpace {
    pub type_id : TypeId,
    pub terms : Vec<Box<dyn FuncImpl>>
}

impl PrimitiveTypeSpace {
    ///Constructs a new, initially-empty [`PrimitiveTypeSpace`]
    pub fn new(type_id : TypeId) -> PrimitiveTypeSpace {
        PrimitiveTypeSpace {
            type_id,
            terms : Vec::new()
        }
    }
}
