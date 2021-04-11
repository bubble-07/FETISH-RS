use crate::context::*;

pub trait DisplayableWithContext {
    fn display(&self, context : &Context) -> String;
}
