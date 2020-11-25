use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::type_id::*;
use crate::holed_linear_expression::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum HoledApplication {
    FunctionHoled(TermReference, TypeId), //_(x). Second argument is ret type.
    ArgumentHoled(TermPointer)  //f(_)
}

impl HoledApplication {
    pub fn format_string(&self, state : &InterpreterState, filler : String) -> String {
        match (&self) {
            HoledApplication::FunctionHoled(arg_ref, _) => {
                let mut result = filler;
                result += "(";

                let arg_str = arg_ref.display(state);

                result += &arg_str;
                result += ")";
                result
            },
            HoledApplication::ArgumentHoled(func_ptr) => {
                let mut result = func_ptr.display(state);
                result += "(";
                result += &filler;
                result += ")";
                result
            }
        }
    }
    pub fn to_linear_expression(&self) -> HoledLinearExpression {
        let mut chain = Vec::new();
        chain.push(self.clone());
        HoledLinearExpression {
            chain
        }
    }

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
