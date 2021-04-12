use std::collections::HashMap;
use crate::type_id::*;
use crate::primitive_type_space::*;
use crate::func_impl::*;
use crate::params::*;
use crate::primitive_term_pointer::*;

pub struct PrimitiveDirectory {
    pub primitive_type_spaces : HashMap::<TypeId, PrimitiveTypeSpace>
}

pub fn get_default_primitive_directory(type_info_directory : &TypeInfoDirectory) -> PrimitiveDirectory {
    let mut result = PrimitiveDirectory::new(type_info_directory);
    let scalar_type = type_info_directory.get(&Type::VecType(1));
    let vector_type = type_info_directory.get(&Type::VecType(DIM));
    let unary_scalar_func_type = type_info_directory.get(&Type::FuncType(scalar_type, scalar_type));
    let binary_scalar_func_type = type_info_directory.get(&Type::FuncType(scalar_type, unary_scalar_func_type));
    
    let map_impl = MapImpl {
        unary_scalar_func_type,
        vector_type,
        scalar_type
    };

    let fill_impl = FillImpl {
        scalar_type,
        vector_type
    };

    let set_head_impl = SetHeadImpl {
        vector_type,
        scalar_type
    };

    let head_impl = HeadImpl {
        vector_type,
        scalar_type
    };

    let rotate_impl = RotateImpl {
        vector_type
    };

    let reduce_impl = ReduceImpl {
        binary_scalar_func_type,
        scalar_type,
        vector_type
    };

    result.add(Box::new(map_impl), type_info_directory);
    result.add(Box::new(fill_impl), type_info_directory);
    result.add(Box::new(set_head_impl), type_info_directory);
    result.add(Box::new(head_impl), type_info_directory);
    result.add(Box::new(rotate_impl), type_info_directory);
    result.add(Box::new(reduce_impl), type_info_directory);

    //Binary functions
    for type_id in [scalar_type, vector_type].iter() {
        result.add_binary_func(*type_id, Box::new(AddOperator {}), type_info_directory);
        result.add_binary_func(*type_id, Box::new(SubOperator {}), type_info_directory);
        result.add_binary_func(*type_id, Box::new(MulOperator {}), type_info_directory);
    }

    for ret_type in [scalar_type, vector_type].iter() {
        for ignored_type in [scalar_type, vector_type].iter() {
            let const_impl = ConstImpl {
                ret_type : *ret_type,
                ignored_type : *ignored_type
            };
            result.add(Box::new(const_impl), type_info_directory);
        }
    }
    for in_type in [scalar_type, vector_type].iter() {
        for middle_type in [scalar_type, vector_type].iter() {
            for ret_type in [scalar_type, vector_type].iter() {
                let compose_impl = ComposeImpl::new(type_info_directory, *in_type, *middle_type, *ret_type);
                result.add(Box::new(compose_impl), type_info_directory);
            }
        }
    }
    result
}

impl PrimitiveDirectory {

    pub fn get_primitive(&self, primitive_term_pointer : PrimitiveTermPointer) -> &dyn FuncImpl {
        let primitive_type_space = self.primitive_type_spaces.get(&primitive_term_pointer.type_id).unwrap();
        let term = &primitive_type_space.terms[primitive_term_pointer.index];
        term.as_ref()
    }

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
    pub fn add(&mut self, func_impl : Box<dyn FuncImpl>, type_info_directory : &TypeInfoDirectory) {
        let func_type = func_impl.func_type(type_info_directory);
        let primitive_type_space = self.primitive_type_spaces.get_mut(&func_type).unwrap();
        primitive_type_space.terms.push(func_impl);
    }

    pub fn add_binary_func(&mut self, type_id : TypeId, binary_func : Box<dyn BinaryArrayOperator>, 
                                      type_info_directory : &TypeInfoDirectory) {
        let binary_func_impl = BinaryFuncImpl {
            elem_type : type_id,
            f : binary_func
        };
        self.add(Box::new(binary_func_impl), type_info_directory);
    }
}
