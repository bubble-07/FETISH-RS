use crate::type_ids::*;
use std::collections::HashMap;

pub struct ApplicationTable {
    func_space : TypeId,
    arg_space : TypeId,
    result_space : TypeId,
    table : HashMap::<(usize, usize), usize>
}
