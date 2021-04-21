#[derive(Clone, Copy, PartialEq, Hash, Eq)]
pub enum TermIndex {
    Primitive(usize),
    NonPrimitive(usize) 
}
