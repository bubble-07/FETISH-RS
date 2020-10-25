use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::type_id::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum HoledApplication {
    FunctionHoled(TermReference, TypeId), //_(x). Second argument is ret type.
    ArgumentHoled(TermPointer)  //f(_)
}

impl HoledApplication {
    pub fn cap(&self, cap : TermReference) -> TermApplication {
        match (&self) {
            HoledApplication::FunctionHoled(arg_term, ret_type) => {
                match (cap) {
                    TermReference::VecRef(_) => { panic!(); },
                    TermReference::FuncRef(func_ptr) => TermApplication {
                        func_ptr : func_ptr,
                        arg_ref : arg_term.clone()
                    }
                }
            },
            HoledApplication::ArgumentHoled(func_ptr) => TermApplication {
                func_ptr : func_ptr.clone(),
                arg_ref : cap
            }
        }
    }
    pub fn get_hole_type(&self) -> TypeId {
        match (&self) {
            HoledApplication::FunctionHoled(arg_term, ret_type) => {
                let func_type = Type::FuncType(arg_term.get_type(), *ret_type);
                get_type_id(&func_type)
            },
            HoledApplication::ArgumentHoled(func_ptr) => {
                match (get_type(func_ptr.type_id)) {
                    Type::VecType(_) => { panic!(); },
                    Type::FuncType(arg_type, _) => arg_type
                }
            }
        }
    }
    pub fn get_type(&self) -> TypeId {
        match (&self) {
            HoledApplication::FunctionHoled(_, ret_type) => *ret_type,
            HoledApplication::ArgumentHoled(func_ptr) => {
                match (get_type(func_ptr.type_id)) { 
                    Type::VecType(_) => { panic!(); },
                    Type::FuncType(_, ret_type) => ret_type
                }
            }
        }
    }
}
