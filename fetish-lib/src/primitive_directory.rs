use std::collections::HashMap;
use crate::type_id::*;
use crate::primitive_type_space::*;
use crate::func_impl::*;
use crate::params::*;
use crate::primitive_term_pointer::*;

///A directory of primitive function terms ([`FuncImpl`]s),
///consisting of one [`PrimitiveTypeSpace`] for each function [`TypeId`]
///in some [`TypeInfoDirectory`].
pub struct PrimitiveDirectory {
    pub primitive_type_spaces : HashMap::<TypeId, PrimitiveTypeSpace>
}

impl PrimitiveDirectory {
    ///Given a [`PrimitiveTermPointer`] pointing to a primitive term in this
    ///[`PrimitiveDirectory`], yields the primitive term as a [`FuncImpl`].
    pub fn get_primitive(&self, primitive_term_pointer : PrimitiveTermPointer) -> &dyn FuncImpl {
        let primitive_type_space = self.primitive_type_spaces.get(&primitive_term_pointer.type_id).unwrap();
        let term = &primitive_type_space.terms[primitive_term_pointer.index];
        term.as_ref()
    }

    ///Constructs a new, initially-empty [`PrimitiveDirectory`].
    pub fn new(type_info_directory : &TypeInfoDirectory) -> PrimitiveDirectory {
        let mut primitive_type_spaces = HashMap::new();

        for type_id in 0..type_info_directory.get_total_num_types() {
            if (!type_info_directory.is_vector_type(type_id)) {
                primitive_type_spaces.insert(type_id, PrimitiveTypeSpace::new(type_id));
            }
        }
        PrimitiveDirectory {
            primitive_type_spaces
        }
    }
    ///Adds the given [`FuncImpl`] to this [`PrimitiveDirectory`], which is assumed
    ///to reference types pulled from the given [`TypeInfoDirectory`].
    pub fn add(&mut self, func_impl : Box<dyn FuncImpl>, type_info_directory : &TypeInfoDirectory) {
        let func_type = func_impl.func_type(type_info_directory);
        let primitive_type_space = self.primitive_type_spaces.get_mut(&func_type).unwrap();
        primitive_type_space.terms.push(func_impl);
    }

    ///Convenient wrapper around [`add`] which allows adding a [`BinaryArrayOperator`]
    ///to this [`PrimitiveDirectory`] whose element type is the given [`TypeId`] within
    ///the given [`TypeInfoDirectory`].
    pub fn add_binary_func(&mut self, type_id : TypeId, binary_func : Box<dyn BinaryArrayOperator>, 
                                      type_info_directory : &TypeInfoDirectory) {
        let binary_func_impl = BinaryFuncImpl {
            elem_type : type_id,
            f : binary_func
        };
        self.add(Box::new(binary_func_impl), type_info_directory);
    }
}
