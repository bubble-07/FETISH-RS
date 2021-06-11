use crate::term_pointer::*;
use crate::type_id::*;
use crate::displayable_with_context::*;
use crate::context::*;

use serde::{Serialize, Deserialize};

///A pointer to a primitive function term ([`crate::func_impl::FuncImpl`])
///within a [`Context`]'s [`crate::primitive_directory::PrimitiveDirectory`].
#[derive(Copy, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub struct PrimitiveTermPointer {
    pub type_id : TypeId,
    pub index : usize
}

impl DisplayableWithContext for PrimitiveTermPointer {
    fn display(&self, ctxt : &Context) -> String {
        let primitive = ctxt.get_primitive(*self);
        primitive.get_name()
    }
}
