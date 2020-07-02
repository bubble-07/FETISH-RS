use crate::interpreter_state::*;

pub trait DisplayableWithState {
    fn display(&self, state : &InterpreterState) -> String;
}
