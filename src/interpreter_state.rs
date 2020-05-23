use std::collections::HashMap;
use crate::type_ids::*;
use crate::application_table::*;
use crate::type_space::*;

pub struct InterpreterState {
    application_tables : HashMap::<TypeId, ApplicationTable>,
    type_spaces : HashMap::<TypeId, TypeSpace>
}
