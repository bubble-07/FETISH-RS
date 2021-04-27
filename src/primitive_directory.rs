use fetish_lib::everything::*;

pub fn get_default_primitive_directory(type_info_directory : &TypeInfoDirectory) -> PrimitiveDirectory {
    let mut result = PrimitiveDirectory::new(type_info_directory);
    let scalar_type = 0 as TypeId;
    let vector_type = 1 as TypeId;
    let unary_scalar_func_type = type_info_directory.get_func_type_id(scalar_type, scalar_type);
    let binary_scalar_func_type = type_info_directory.get_func_type_id(scalar_type, unary_scalar_func_type);
    
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


