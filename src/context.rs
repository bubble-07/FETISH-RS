use fetish_lib::everything::*;
use crate::space_info::*;
use crate::params::*;
use crate::type_id::*;

pub fn get_default_context() -> Context {
    let type_info_directory = get_default_type_info_directory(DIM);
    let space_info_directory = get_default_space_info_directory(&type_info_directory);
    let primitive_directory =  get_default_primitive_directory(&type_info_directory);
    Context {
        type_info_directory,
        space_info_directory,
        primitive_directory
    }
}

