use crate::type_ids::*;
use crate::interpreter_state::*;
use crate::term_pointer::*;
use crate::term::*;

pub trait FuncImpl {
    fn ret_type(&self) -> TypeId;
    fn required_arg_types(&self) -> Vec::<TypeId>;
    fn evaluate(&self, state : &mut InterpreterState, args : Vec::<TermPointer>) -> Term;
}
