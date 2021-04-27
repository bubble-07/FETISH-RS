///An index to a nonprimitive [`crate::term::PartiallyAppliedTerm`] or
///a primitive [`crate::func_impl::FuncImpl`] within an [`InterpreterState`].
#[derive(Clone, Copy, PartialEq, Hash, Eq)]
pub enum TermIndex {
    Primitive(usize),
    NonPrimitive(usize) 
}
