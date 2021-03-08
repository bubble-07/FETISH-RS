use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_id::*;
use std::cmp::*;
use std::fmt::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

#[derive(Clone)]
pub enum TypeAction {
    Applying(TypeId),
    Passing(TypeId)
}

impl TypeAction {
    pub fn get_actions_for(source_type : TypeId, dest_type : TypeId) -> Vec<TypeAction> {
        let mut result = Vec::new();

        if (!is_vector_type(source_type)) {
            let ret_type = get_ret_type_id(source_type);
            if (ret_type == dest_type) {
                let arg_type = get_arg_type_id(source_type);
                let action = TypeAction::Passing(arg_type);
                result.push(action);
            }
        }

        let func_sig = Type::FuncType(source_type, dest_type);
        if (has_type(&func_sig)) {
            let func_type_id = get_type_id(&func_sig);
            let action = TypeAction::Applying(func_type_id);
            result.push(action);
        }

        result
    }
}
