use crate::type_id::*;
use crate::context::*;

#[derive(Clone)]
pub enum TypeAction {
    Applying(TypeId),
    Passing(TypeId)
}

impl TypeAction {
    pub fn get_actions_for(ctxt : &Context, source_type : TypeId, dest_type : TypeId) -> Vec<TypeAction> {
        let mut result = Vec::new();

        if (ctxt.is_vector_type(dest_type)) {
            //Vector destination types are forbidden
            return result;
        }

        if (!ctxt.is_vector_type(source_type)) {
            let ret_type = ctxt.get_ret_type_id(source_type);
            if (ret_type == dest_type) {
                let arg_type = ctxt.get_arg_type_id(source_type);
                let action = TypeAction::Passing(arg_type);
                result.push(action);
            }
        }

        let func_sig = Type::FuncType(source_type, dest_type);
        if (ctxt.has_type(&func_sig)) {
            let func_type_id = ctxt.get_type_id(&func_sig);
            let action = TypeAction::Applying(func_type_id);
            result.push(action);
        }

        result
    }
}
