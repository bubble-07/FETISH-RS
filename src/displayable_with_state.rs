use crate::interpreter_state::*;

///Trait for things that may be rendered as a [`String`] provided
///that an [`InterpreterState`] is available.
pub trait DisplayableWithState {
    fn display(&self, state : &InterpreterState) -> String;
}
