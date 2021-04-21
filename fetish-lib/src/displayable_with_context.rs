use crate::context::*;

///Trait for things that may be rendered as a [`String`] provided
///that a [`Context`] is available.
pub trait DisplayableWithContext {
    fn display(&self, context : &Context) -> String;
}
